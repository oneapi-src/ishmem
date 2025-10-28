/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "ipc.h"
#include "memory.h"
#include "runtime.h"
#include "accelerator.h"
#include <thread>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <level_zero/zet_api.h>
#include <poll.h>

#define CPU_RELAX() asm volatile("rep; nop")

#define ISHMEMI_PIDFD_SUPPORT 1

/* syscall IDs for pidfd exchange */
#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434 /* syscall ID for most architectures */
#else
#if __NR_pidfd_open != 434
#warning "Possible conflict with syscall id for pidfd_open on this system"
#define ISHMEMI_PIDFD_SUPPORT 0
#endif
#endif

#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438 /* syscall ID for most architectures */
#else
#if __NR_pidfd_getfd != 438
#warning "Possible conflict with syscall id for pidfd_getfd on this system"
#define ISHMEMI_PIDFD_SUPPORT 0
#endif
#endif

#if ISHMEMI_PIDFD_SUPPORT == 0
#warning "Disabling support for pidfd"
#endif

/* Function pointer definitions for IPC functions when Implicit Scaling is enabled */
typedef ze_result_t (*fn_zexMemGetIpcHandles)(ze_context_handle_t, const void *, uint32_t *,
                                              ze_ipc_mem_handle_t *);
typedef ze_result_t (*fn_zexMemOpenIpcHandles)(ze_context_handle_t, ze_device_handle_t, uint32_t,
                                               ze_ipc_mem_handle_t *, ze_ipc_memory_flags_t,
                                               void **);

fn_zexMemGetIpcHandles zexMemGetIpcHandles;
fn_zexMemOpenIpcHandles zexMemOpenIpcHandles;
bool ishmemi_only_intra_node = false;

/* Number of attempts to recv IPC handle from a remote rank */
#define MAX_IPC_RETRIES 5

#define SOCK_CHECK(call)                                                                           \
    do {                                                                                           \
        int sock_err = call;                                                                       \
        if (sock_err < 0) {                                                                        \
            ISHMEM_DEBUG_MSG("SOCK FAIL: call = '%s' result = '%d', errno %d (%s)\n", #call,       \
                             sock_err, errno, strerror(errno));                                    \
            ret = sock_err;                                                                        \
        }                                                                                          \
    } while (0)

#define SYSCALL_CHECK(call)                                                                        \
    do {                                                                                           \
        int syscall_err = call;                                                                    \
        if (syscall_err < 0) {                                                                     \
            ISHMEM_DEBUG_MSG("SYSCALL FAIL: call = '%s' result = '%d', errno %d (%s)\n", #call,    \
                             syscall_err, errno, strerror(errno));                                 \
            ret = syscall_err;                                                                     \
        }                                                                                          \
    } while (0)

/* This is defined as the max socket name length sun_path in sockaddr_un */
#define SOCK_MAX_STR_LEN 108

static int local_rank, local_size;
static int responder_ready;
static std::thread responder_thread;

static struct ipc_data_t {
    pid_t pid;
    int local_rank;
    int local_size;
    int ipc_fd[2];
    ze_ipc_mem_handle_t ipc_handle[2];
    int nfds;
} ipc_data;

/* marked static because this is debug code only used in this source file */
static void ishmemi_printfd(const char *prefix, int fd)
{
    struct stat sb;
    int ret = fstat(fd, &sb);
    if (ret != 0)
        ISHMEM_DEBUG_MSG("%s fstat %d returns errno %d(%s)\n", prefix, fd, errno, strerror(errno));
    else {
        ISHMEM_DEBUG_MSG(
            "%s: st_dev: %lu st_ino %lu st_mode 0%o st_nlink %lu st_uid %d st_gid %d "
            "st_rdev %lu st_size %ld st_blksize %ld st_blocks %ld \n",
            prefix, sb.st_dev, sb.st_ino, sb.st_mode, sb.st_nlink, sb.st_uid, sb.st_gid, sb.st_rdev,
            sb.st_size, sb.st_blksize, sb.st_blocks);
    }
}

/* IPC exchange implementations */
static int ipc_init_pidfd();
static int ipc_init_sockets();

/* TODO: Update */
/* IPC setup steps
 * 1) exchange PIDs of all local ranks, using omeshmemi_runtime_node_fcollect
 *    note, this won't work for supernodes larger than TEAM_NODE
 * 2) create thread to respond to connections by sending our ipchandle
 * 5) barrier
 * 6) loop through other nodes, connecting and receiving response
 * 7) join the responding thread
 */

int ishmemi_ipc_init()
{
    int ret = 0;
    ze_ipc_mem_handle_t ipc_handle[2];
    uint32_t nfds = 0;
    bool zex_passed = false;
    ze_result_t ze_ret;

    ::memset(&ipc_handle, 0, sizeof(ze_ipc_mem_handle_t) * 2);

    local_rank = ishmemi_runtime->get_node_rank(ishmemi_runtime->get_rank());
    local_size = ishmemi_runtime->get_node_size();
    ISHMEM_CHECK_GOTO_MSG((local_size > MAX_LOCAL_PES), fn_fail, "get_node_size > MAX_LOCAL_PES\n");
    ISHMEM_DEBUG_MSG("we are local rank %d of %d\n", local_rank, local_size);

    /* Check if IPC functions for implicit scaling are present */
    ze_ret = zeDriverGetExtensionFunctionAddress(ishmemi_gpu_driver, "zexMemGetIpcHandles",
                                                 (void **) &zexMemGetIpcHandles);
    if (ze_ret != ZE_RESULT_SUCCESS) zexMemGetIpcHandles = NULL;

    ze_ret = zeDriverGetExtensionFunctionAddress(ishmemi_gpu_driver, "zexMemOpenIpcHandles",
                                                 (void **) &zexMemOpenIpcHandles);
    if (ze_ret != ZE_RESULT_SUCCESS) zexMemOpenIpcHandles = NULL;

    /* Create IPC handle for the heap */
    if (zexMemGetIpcHandles) {
        /* Get the number of IPC handles (i.e. check if Implicit Scaling is turned on) */
        ZE_CHECK(zexMemGetIpcHandles(ishmemi_ze_context, ishmemi_heap_base, &nfds, NULL));

        /* Make sure that nfds is set correctly before proceeding */
        if (nfds > 0) {
            /* Create the IPC handle(s) */
            ZE_CHECK(zexMemGetIpcHandles(ishmemi_ze_context, ishmemi_heap_base, &nfds, ipc_handle));
            zex_passed = true;
        }
    }

    if (!zex_passed) {
        /* Use regular API for IPC handle creation */
        ZE_CHECK(zeMemGetIpcHandle(ishmemi_ze_context, ishmemi_heap_base, &ipc_handle[0]));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        nfds = 1;
    }

    /* Copy our info to ipc_data to be used in the IPC setup functions */
    ipc_data.pid = getpid();
    ipc_data.local_rank = local_rank;
    ipc_data.local_size = local_size;
    ipc_data.nfds = static_cast<int>(nfds);

    ipc_data.ipc_fd[0] = ipc_data.ipc_fd[1] = -1;

    for (size_t i = 0; i < nfds; ++i) {
        memcpy(&ipc_data.ipc_fd[i], &ipc_handle[i], sizeof(int));
        memcpy(&ipc_data.ipc_handle[i], &ipc_handle[i], sizeof(ze_ipc_mem_handle_t));
    }

    /* First attempt pidfd if enabled */
    if (ishmemi_params.ENABLE_GPU_IPC_PIDFD) {
        ret = ipc_init_pidfd();
    } else {
        ret = -1;
    }

    if (ret != 0) {
        if (ishmemi_params.ENABLE_GPU_IPC_PIDFD) {
            ISHMEM_DEBUG_MSG("IPC init with PIDFD failed '%d', falling back to sockets\n", ret);
        }

        /* pidfd is not supported, so fallback to sockets implementation */
        ret = ipc_init_sockets();
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "IPC init with sockets failed '%d'\n", ret);
    }

    /* Initialize the local ipc_buffer info */
    ishmemi_mmap_gpu_info->ipc_buffer_delta[local_rank + 1] = (ptrdiff_t) 0;
    ishmemi_ipc_buffer_delta[local_rank + 1] = (ptrdiff_t) 0;
    ishmemi_ipc_buffers[local_rank + 1] = ishmemi_heap_base;
    ISHMEM_DEBUG_MSG("ipc_buffer[%d] = %p\n", local_rank + 1, ishmemi_heap_base);

    /* Populate local_pes in info */
    for (int i = 0; i < ishmemi_cpu_info->n_pes; ++i) {
        /* Note: local_pes[i] == 0 means "not local" */
        int local_idx = ishmemi_runtime->get_node_rank(i);
        if (local_idx == -1) {
            ishmemi_mmap_gpu_info->local_pes[i] = 0; /* For device use */
            ishmemi_local_pes[i] = 0;                /* For hose use */
            ISHMEM_DEBUG_MSG("local_pes[%d] = NA\n", i);
        } else {
            /* Validate local_idx */
            ISHMEM_CHECK_GOTO_MSG(local_idx > MAX_LOCAL_PES, fn_fail,
                                  "maximum local pe index is %d, found local pe index %d\n",
                                  MAX_LOCAL_PES, local_idx);

            /* Skips index 0 */
            ishmemi_mmap_gpu_info->local_pes[i] =
                static_cast<uint8_t>(local_idx + 1);                    /* For device use */
            ishmemi_local_pes[i] = static_cast<uint8_t>(local_idx + 1); /* For host use */
            ISHMEM_DEBUG_MSG("local_pes[%d] = %d\n", i, local_idx + 1);
        }
    }

    ishmemi_only_intra_node = (local_size == ishmemi_n_pes);
    ishmemi_mmap_gpu_info->only_intra_node = ishmemi_only_intra_node;
    ishmemi_cpu_info->use_ipc = true;

fn_exit:
    return ret;
fn_fail:
    goto fn_exit;
}

int ishmemi_ipc_fini()
{
    int ret = 0;

    /* Close IPC handles */
    for (int i = 0; i < local_size; ++i) {
        /* This loop skips the local symmetric heap since it does not correspond to an IPC handle */
        if (i == local_rank) continue;
        ZE_CHECK(zeMemCloseIpcHandle(ishmemi_ze_context, ishmemi_ipc_buffers[i + 1]));
        /* ret could be non-zero, but continue attempting to close all the other IPC handles */
    }

    /* Symmetric heap is freed by memory_fini */
    /* Assumes no kernels are running when calling ipc_fini */
    ishmemi_runtime->node_barrier();

    ishmemi_cpu_info->use_ipc = false;

    return ret;
}

/* pidfd-based IPC handle exchange implementation */
static int ipc_init_pidfd()
{
    int ret = 0;
    int fd = -1, pidfd = -1, dupfd[2];
    char file_template[] = "/tmp/ishmem-ipc-check-capability-XXXXXX";
    char *temp_file = file_template;
    void *temp_ipc_buffer;
    ipc_data_t *local_heap_data = NULL, *heap_data = NULL, *local_data = NULL;
    ze_ipc_mem_handle_t remote_ipc_handle[2];

    dupfd[0] = dupfd[1] = -1;

    /* Validate that pidfd functions are present */
    {
        fd = mkstemp(temp_file);
        ISHMEMI_CHECK_RESULT((fd == -1), 0, fn_fail);

        pidfd = static_cast<int>(syscall(__NR_pidfd_open, ipc_data.pid, 0));
        SYSCALL_CHECK(pidfd);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        dupfd[0] = static_cast<int>(syscall(__NR_pidfd_getfd, pidfd, fd, 0));
        SYSCALL_CHECK(dupfd[0]);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        /* Cleanup above fds */
        close(dupfd[0]);
        close(pidfd);
        close(fd);
        unlink(temp_file);
        dupfd[0] = pidfd = fd = -1;
    }

    /* Attempt to use pidfd to communicate IPC handles */
    /* Allocate pid arrays to communicate across PEs */
    heap_data = (ipc_data_t *) ishmemi_runtime->calloc(MAX_LOCAL_PES, sizeof(ipc_data_t));
    ISHMEM_CHECK_GOTO_MSG((heap_data == NULL), fn_fail, "unable to allocate heap_data\n");

    local_heap_data = (ipc_data_t *) ishmemi_runtime->calloc(1, sizeof(ipc_data_t));
    ISHMEM_CHECK_GOTO_MSG((local_heap_data == NULL), fn_fail,
                          "unable to allocate local_heap_data\n");

    local_data = (ipc_data_t *) ::malloc(MAX_LOCAL_PES * sizeof(ipc_data_t));
    ISHMEM_CHECK_GOTO_MSG((local_data == NULL), fn_fail, "unable to allocate local_data\n");

    /* Copy our info to the host symmetric heap for use in fcollect */
    memcpy(&local_heap_data[0], &ipc_data, sizeof(ipc_data_t));

    ishmemi_runtime->node_barrier();

    /* Gather the info from other PEs */
    ishmemi_runtime->node_fcollect(heap_data, local_heap_data, sizeof(ipc_data_t));
    ishmemi_copy(local_data, heap_data, static_cast<size_t>(local_size) * sizeof(ipc_data_t));

    /* Validate that every pid can be opened locally */
    for (int i = 0; i < local_size; ++i) {
        if (i == local_rank) continue;

        /* Open a file descriptor for the PE using its pid */
        pidfd = static_cast<int>(syscall(__NR_pidfd_open, local_data[i].pid, 0));
        SYSCALL_CHECK(pidfd);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        /* Duplicate the IPC file descriptor into the calling process */
        /* Note, this call might fail on some systems and cause fallback to the sockets method, even
         * if it succeeded above. For example, on Ubuntu, CAP_SYS_PTRACE is not enabled by default,
         * which is necessary for this call to work on a remote pidfd and fd. */
        for (int j = 0; j < local_data[i].nfds; ++j) {
            dupfd[j] =
                static_cast<int>(syscall(__NR_pidfd_getfd, pidfd, local_data[i].ipc_fd[j], 0));
            SYSCALL_CHECK(dupfd[j]);
            ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
        }

        /* Close pidfd as it is no longer needed */
        close(pidfd);
        pidfd = -1;

        /* Build the remote IPC handle */
        ::memset(&remote_ipc_handle, 0, sizeof(ze_ipc_mem_handle_t) * 2);

        for (int j = 0; j < local_data[i].nfds; ++j) {
            memcpy(&remote_ipc_handle[j], &local_data[i].ipc_handle[j],
                   sizeof(ze_ipc_mem_handle_t));
            memcpy(&remote_ipc_handle[j], &dupfd[j], sizeof(int));
        }

        /* Open the IPC handle */
        if (zexMemOpenIpcHandles) {
            ZE_CHECK(zexMemOpenIpcHandles(ishmemi_ze_context, ishmemi_gpu_device,
                                          static_cast<uint32_t>(local_data[i].nfds),
                                          remote_ipc_handle, 0, &temp_ipc_buffer));
        } else {
            ZE_CHECK(zeMemOpenIpcHandle(ishmemi_ze_context, ishmemi_gpu_device,
                                        remote_ipc_handle[0], 0, &temp_ipc_buffer));
        }
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        ISHMEM_DEBUG_MSG("ipc_buffer[%d] = %p\n", i + 1, temp_ipc_buffer);

        /* Store the delta between heap bases to minimize adjustment later */
        ishmemi_mmap_gpu_info->ipc_buffer_delta[i + 1] =
            ((ptrdiff_t) temp_ipc_buffer - (ptrdiff_t) ishmemi_heap_base);
        ishmemi_ipc_buffer_delta[i + 1] =
            ((ptrdiff_t) temp_ipc_buffer - (ptrdiff_t) ishmemi_heap_base);
        ishmemi_ipc_buffers[i + 1] = temp_ipc_buffer;

        /* Close dupfd as it is no longer needed */
        for (int j = 0; j < local_data[i].nfds; ++j) {
            close(dupfd[j]);
            dupfd[j] = -1;
        }
    }

fn_exit:
    /* TODO: Gather status from every PE */

    /* Cleanup heap_pids */
    ISHMEMI_FREE(ishmemi_runtime->free, local_heap_data);
    ISHMEMI_FREE(ishmemi_runtime->free, heap_data);
    ISHMEMI_FREE(::free, local_data);

    return ret;
fn_fail:
    /* pidfd mechanism isn't supported */
    /* Cleanup any remaining fds */
    if (dupfd[0] != -1) close(dupfd[0]);
    if (dupfd[1] != -1) close(dupfd[1]);
    if (pidfd != -1) close(pidfd);
    if (fd != -1) {
        close(fd);
        unlink(temp_file);
    }

    goto fn_exit;
}

/* Socket-based IPC handle exchange implementation */
static void socket_send_ipc_handle(void *arg);
static int socket_recv_ipc_handle(int pe, pid_t pe_pid, ze_ipc_mem_handle_t (&handle)[2],
                                  int (&fd)[2]);

static int ipc_init_sockets()
{
    int ret = 0, fini = 0;
    pid_t *local_heap_pid = nullptr, *heap_pids = nullptr, *local_pids = nullptr;
    void *temp_ipc_buffer;
    ze_ipc_mem_handle_t remote_ipc_handle[2];

    /* Allocate pid arrays to communicate accross PEs */
    heap_pids = (pid_t *) ishmemi_runtime->calloc(MAX_LOCAL_PES, sizeof(pid_t));
    ISHMEM_CHECK_GOTO_MSG((heap_pids == NULL), fn_fail, "unable to allocate heap_pids\n");

    local_heap_pid = (pid_t *) ishmemi_runtime->calloc(1, sizeof(pid_t));
    ISHMEM_CHECK_GOTO_MSG((local_heap_pid == NULL), fn_fail, "unable to allocate local_heap_pid\n");

    local_pids = (pid_t *) ::malloc(MAX_LOCAL_PES * sizeof(pid_t));
    ISHMEM_CHECK_GOTO_MSG((local_pids == NULL), fn_fail, "unable to allocate local_pids\n");

    /* Initialize local_pids and ishmemi_ipc_buffers */
    for (int i = 0; i < MAX_LOCAL_PES; ++i) {
        local_pids[i] = -1;
    }
    for (int i = 0; i <= MAX_LOCAL_PES; ++i) {
        ishmemi_ipc_buffers[i] = nullptr;
    }

    /* Copy our pid to the host symmetric heap for use in fcollect */
    local_heap_pid[0] = ipc_data.pid;

    ishmemi_runtime->node_barrier();

    /* Gather pids from other PEs */
    ishmemi_runtime->node_fcollect(heap_pids, local_heap_pid, sizeof(pid_t));
    ishmemi_copy(local_pids, heap_pids, static_cast<size_t>(local_size) * sizeof(pid_t));

    for (int i = 0; i < local_size; ++i) {
        ISHMEM_DEBUG_MSG("heap_pids[%d] = %d (%d)\n", i, heap_pids[i], local_pids[i]);
    }

    /* Wait for all processes to copy before proceeding */
    ishmemi_runtime->node_barrier();

    /* Prepare for receiving buffers from processes */
    /* fork responder */
    responder_ready = false;
    responder_thread = std::thread(socket_send_ipc_handle, nullptr);

    while (!responder_ready)
        CPU_RELAX();

    /* this barrier is to assure all local ranks have responders up and running */
    ishmemi_runtime->node_barrier();

    /* loop through all local pe's except us fetching ipc handles */
    for (int i = 0; i < local_size; ++i) {
        int nfds = 1;
        int ipc_fd[2];
        if (i == local_rank) continue;

        /* Attempt to get the ipc handle from the socket message */
        for (int j = 0; j < MAX_IPC_RETRIES; ++j) {
            ret = socket_recv_ipc_handle(i, local_pids[i], remote_ipc_handle, ipc_fd);

            ISHMEM_DEBUG_MSG("ipc handle for local pe %d is (%d, %d)\n", i, ipc_fd[0], ipc_fd[1]);
            ishmemi_printfd("received", ipc_fd[0]);
            if (ipc_fd[1] != -1) {
                nfds += 1;
                ishmemi_printfd("received", ipc_fd[1]);
            }
            if (ipc_fd[0] == -1) {
                ISHMEM_DEBUG_MSG("socket_recv_ipc_handle for local PE %d returned -1\n", i);
                continue;
            }
            break;
        }

        /* Check that we have a valid file descriptor for the remote IPC handle */
        ISHMEMI_CHECK_RESULT((ipc_fd[0] == -1), 0, fn_fail);

        /* Build the remote IPC handle */
        for (int i = 0; i < nfds; ++i) {
            memcpy(&remote_ipc_handle[i], &ipc_fd[i], sizeof(ipc_fd));
        }

        /* Open the IPC handle */
        if (zexMemOpenIpcHandles) {
            ZE_CHECK(zexMemOpenIpcHandles(ishmemi_ze_context, ishmemi_gpu_device,
                                          static_cast<uint32_t>(nfds), remote_ipc_handle, 0,
                                          &temp_ipc_buffer));
        } else {
            ZE_CHECK(zeMemOpenIpcHandle(ishmemi_ze_context, ishmemi_gpu_device,
                                        remote_ipc_handle[0], 0, &temp_ipc_buffer));
        }
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        ISHMEM_DEBUG_MSG("ipc_buffer[%d] = %p\n", i + 1, temp_ipc_buffer);

        /* Store the delta between heap bases to minimize adjustment later */
        ishmemi_mmap_gpu_info->ipc_buffer_delta[i + 1] =
            ((ptrdiff_t) temp_ipc_buffer - (ptrdiff_t) ishmemi_heap_base);
        ishmemi_ipc_buffer_delta[i + 1] =
            ((ptrdiff_t) temp_ipc_buffer - (ptrdiff_t) ishmemi_heap_base);
        ishmemi_ipc_buffers[i + 1] = temp_ipc_buffer;
    }

fn_exit:
    /* Barrier to ensure all local PEs have completed init (pass or fail) */
    ishmemi_runtime->node_barrier();

    /* Cleanup heap_pids */
    ISHMEMI_FREE(ishmemi_runtime->free, local_heap_pid);
    ISHMEMI_FREE(ishmemi_runtime->free, heap_pids);
    ISHMEMI_FREE(::free, local_pids);

    /* Complete the responder thread */
    responder_ready = false;
    responder_thread.join();

    return ret;
fn_fail:
    fini = ishmemi_ipc_fini();
    if (fini != 0) ret = fini;
    goto fn_exit;
}

static void socket_send_ipc_handle(void *arg)
{
    char sock_name[SOCK_MAX_STR_LEN];
    struct sockaddr_un sockaddr;
    socklen_t len = sizeof(sockaddr);
    int response_socket = -1;
    int ret = 0;
    int connfd = -1;

    /* set up and bind listener socket */
    ::memset(&sockaddr, 0, sizeof(sockaddr));

    /* Create the local socket name */
    snprintf(sock_name, SOCK_MAX_STR_LEN, "/tmp/ishmem-ipc-fd-sock-%d:%d", ipc_data.pid,
             ipc_data.local_rank);
    ISHMEM_DEBUG_MSG("responder listening on %s\n", sock_name);
    unlink(sock_name);

    /* Create a socket for local rank */
    response_socket = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    SOCK_CHECK(response_socket);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    sockaddr.sun_family = AF_UNIX;
    strcpy(sockaddr.sun_path, sock_name);

    SOCK_CHECK(bind(response_socket, (struct sockaddr *) &sockaddr, len));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    /* Listen to the socket to accept a connection to the other process */
    SOCK_CHECK(listen(response_socket, MAX_LOCAL_PES));
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

    responder_ready = true;
    while (responder_ready) {
        struct pollfd fds[1];
        fds[0].fd = response_socket;
        fds[0].events = 0;
        fds[0].revents = 0;
        ret = poll(fds, 1, 100);
        if (ret < 0) {
            ISHMEM_DEBUG_MSG("poll returned with error. errno %d(%s)\n", errno, strerror(errno));
            goto fn_fail;
        }
        if (fds[0].revents & POLLNVAL) {
            ISHMEM_DEBUG_MSG("poll performed with invalid request\n");
            goto fn_fail;
        }

        connfd = accept(response_socket, NULL, 0); /* 100 millisecond timeout */
        if (connfd < 0) {
            if (errno == EAGAIN) continue;
            if (errno == EWOULDBLOCK) continue;
            ISHMEM_DEBUG_MSG("accept returned with error. errno %d(%s)\n", errno, strerror(errno));
        }

        SOCK_CHECK(connfd);
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);

        /* Send message to remote PE */
        {
            struct mmsghdr mmsg[2];
            struct cmsghdr *cmsg[2];
            struct iovec iov[2];
            char ctrl_buf[CMSG_SPACE(sizeof(int))][2];
            ishmemi_socket_payload_t payload[2];

            ::memset(mmsg, 0, sizeof(mmsghdr) * 2);
            ::memset(iov, 0, sizeof(iovec) * 2);
            ::memset(ctrl_buf, 0, CMSG_SPACE(sizeof(int)) * 2);
            ::memset(payload, 0, sizeof(ishmemi_socket_payload_t) * 2);

            for (int i = 0; i < ipc_data.nfds; ++i) {
                /* Setup the payload with device and msg-matching info */
                payload[i].src_pe = local_rank;
                memcpy(&payload[i].handle, &ipc_data.ipc_handle[i], sizeof(ze_ipc_mem_handle_t));

                iov[i].iov_base = &payload[i];
                iov[i].iov_len = sizeof(ishmemi_socket_payload_t);

                /* Setup the message header */
                mmsg[i].msg_hdr.msg_control = ctrl_buf[i];
                mmsg[i].msg_hdr.msg_controllen = CMSG_LEN(sizeof(int));
                mmsg[i].msg_hdr.msg_iov = &iov[i];
                mmsg[i].msg_hdr.msg_iovlen = 1;

                /* Setup the control message, which includes the fd */
                cmsg[i] = CMSG_FIRSTHDR(&mmsg[i].msg_hdr);
                cmsg[i]->cmsg_len = CMSG_LEN(sizeof(int));
                cmsg[i]->cmsg_level = SOL_SOCKET;
                cmsg[i]->cmsg_type = SCM_RIGHTS;
                memcpy(CMSG_DATA(cmsg[i]), &ipc_data.ipc_fd[i], sizeof(int));
            }

            /* Send the message */
            SOCK_CHECK(sendmmsg(connfd, mmsg, 2, 0));

            close(connfd);
            connfd = -1;
        }
    }

fn_fail:
    if (response_socket != -1) {
        close(response_socket);
        response_socket = -1;
    }
    if (connfd != -1) {
        close(connfd);
        connfd = -1;
    }
}

static int socket_recv_ipc_handle(int pe, pid_t pe_pid, ze_ipc_mem_handle_t (&handle)[2],
                                  int (&fd)[2])
{
    int ret = 0;
    int query_socket = -1;
    socklen_t remote_sockaddr_len = sizeof(struct sockaddr_un);
    bool query_socket_open = false;
    struct sockaddr_un remote_sockaddr;

    fd[0] = fd[1] = -1;

    /* Create a socket */
    query_socket = socket(AF_UNIX, SOCK_STREAM, 0);
    SOCK_CHECK(query_socket);
    ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    query_socket_open = true;

    /* Connect to remote socket for local rank "pe" */
    {
        char remote_sock_name[SOCK_MAX_STR_LEN];

        ::memset(&remote_sockaddr, 0, sizeof(remote_sockaddr));
        remote_sockaddr.sun_family = AF_UNIX;
        snprintf(remote_sock_name, SOCK_MAX_STR_LEN, "/tmp/ishmem-ipc-fd-sock-%d:%d", pe_pid, pe);

        strcpy(remote_sockaddr.sun_path, remote_sock_name);

        SOCK_CHECK(
            connect(query_socket, (struct sockaddr *) &remote_sockaddr, remote_sockaddr_len));
        ISHMEMI_CHECK_RESULT(ret, 0, fn_fail);
    }

    /* Attempt to recv the message from local rank "pe" */
    {
        struct mmsghdr mmsg[2];
        struct cmsghdr *cmsg;
        struct iovec iov[2];
        char ctrl_buf[CMSG_SPACE(sizeof(int))][2];
        ishmemi_socket_payload_t payload[2];

        ::memset(mmsg, 0, sizeof(mmsghdr) * 2);
        ::memset(iov, 0, sizeof(iovec) * 2);
        ::memset(ctrl_buf, 0, CMSG_SPACE(sizeof(int)) * 2);
        ::memset(payload, 0, sizeof(ishmemi_socket_payload_t) * 2);

        /* Always assume 2 messages are coming, but the data in the second one may not be needed */
        for (int i = 0; i < 2; ++i) {
            iov[i].iov_base = &payload[i];
            iov[i].iov_len = sizeof(ishmemi_socket_payload_t);

            /* Setup the message header */
            mmsg[i].msg_hdr.msg_control = ctrl_buf[i];
            mmsg[i].msg_hdr.msg_controllen = CMSG_LEN(sizeof(int));
            mmsg[i].msg_hdr.msg_iov = &iov[i];
            mmsg[i].msg_hdr.msg_iovlen = 1;
        }

        /* Try to recv the message */
        SOCK_CHECK(recvmmsg(query_socket, mmsg, 2, 0, NULL));
        ISHMEM_CHECK_GOTO_MSG(ret, fn_fail, "%s remote pe %d remote pid %d failed\n", __FUNCTION__,
                              pe, (int) pe_pid);

        /* Check if the message(s) were truncated and fail if so */
        for (int i = 0; i < 2; ++i) {
            if ((mmsg[i].msg_hdr.msg_flags & MSG_TRUNC) ||
                (mmsg[i].msg_hdr.msg_flags & MSG_CTRUNC)) {
                ISHMEM_DEBUG_MSG("Transmission Issue: message truncated (%d)\n",
                                 mmsg[i].msg_hdr.msg_flags);
                goto fn_fail;
            }

            /* Pull out the FD from the control message */
            for (cmsg = CMSG_FIRSTHDR(&mmsg[i].msg_hdr); cmsg != nullptr;
                 cmsg = CMSG_NXTHDR(&mmsg[i].msg_hdr, cmsg)) {
                if (cmsg->cmsg_len == CMSG_LEN(sizeof(int)) && cmsg->cmsg_level == SOL_SOCKET &&
                    cmsg->cmsg_type == SCM_RIGHTS) {
                    memcpy(&fd[i], CMSG_DATA(cmsg), sizeof(int));
                    break;
                }
            }
        }

        /* Copy the ipc handle from the remote PE */
        for (int i = 0; i < 2; ++i) {
            memcpy(&handle[i], &payload[i].handle, sizeof(ze_ipc_mem_handle_t));
        }
    }

fn_exit:
    /* Close the socket */
    if (query_socket_open) close(query_socket);

    return ret;
fn_fail:
    goto fn_exit;
}
