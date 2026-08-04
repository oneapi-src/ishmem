#include "ishmemx.h"
#include "ishmem/err.h"
#include "runtime.h"
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <string>

namespace {
    constexpr const char *ISHMEMI_DEFAULT_MPCP_SERVER_PORT = "35555";
}

int ishmemx_get_uniqueid(ishmemx_uniqueid_t *uid)
{
    if (uid == nullptr) {
        ISHMEM_ERROR_MSG("Error! uid is nullptr!\n");
        return -1;
    }

    const char *rank_str = getenv("I_MPI_MPCP_SERVER_NAME");
    if (rank_str == nullptr || rank_str[0] == '\0') {
        // Try to get MASTER_ADDR as fallback
        rank_str = getenv("MASTER_ADDR");
        if (rank_str == nullptr || rank_str[0] == '\0') {
            ISHMEM_ERROR_MSG(
                "I_MPI_MPCP_SERVER_NAME or MASTER_ADDR environment variable not set for MPCP "
                "initialization\n");
            return -1;
        }
    }

    // Server name is expected as host only; port is appended explicitly below.
    if (strchr(rank_str, ':') != nullptr) {
        ISHMEM_ERROR_MSG(
            "I_MPI_MPCP_SERVER_NAME (or MASTER_ADDR fallback) must be host only, without port\n");
        return -1;
    }

    const char *master_port = getenv("MASTER_PORT");
    const char *mpcp_port = getenv("I_MPI_MPCP_SERVER_PORT");
    if (mpcp_port == nullptr || mpcp_port[0] == '\0') {
        mpcp_port = ISHMEMI_DEFAULT_MPCP_SERVER_PORT;
    }
    // It seems MPCP need a different port than MASTER_PORT.
    if (master_port != nullptr && master_port[0] != '\0' && strcmp(mpcp_port, master_port) == 0) {
        ISHMEM_ERROR_MSG("I_MPI_MPCP_SERVER_PORT and MASTER_PORT cannot use the same port\n");
        return -1;
    }

    // Compose MPCP uid in host:port format for runtime consumption.
    std::string uid_str(rank_str);
    uid_str += ":";
    uid_str += mpcp_port;

    size_t copy_len = std::min(sizeof(uid->uniqueid) - 1, uid_str.size());
    strncpy(uid->uniqueid, uid_str.c_str(), copy_len);
    uid->uniqueid[copy_len] = '\0';

    if (uid_str.size() >= sizeof(uid->uniqueid)) {
        ISHMEM_ERROR_MSG("MPCP UID is too long after formatting as host:port\n");
        return -1;
    }

    return 0;
}
