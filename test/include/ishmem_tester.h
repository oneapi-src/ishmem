/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_TESTER_H
#define ISHMEM_TESTER_H

#include <type_traits>  // needed for the enum iterator
#include <ishmem.h>
#include <ishmemx.h>
#include <getopt.h>
#include "common.h"
#include <unistd.h>
#include <map>
#include <ishmem/types.h>
#include <ishmem/err.h>

constexpr size_t x_size = 16;
constexpr size_t y_size = 2;
constexpr size_t z_size = 2;
constexpr size_t num_wg = 4;
constexpr size_t max_wg = 16;

/* Iterator class derived from this question[1] and answer[2] on stackoverflow.com
 * Copyright (C) 2015 Francesco Chemolli[3]
 * Licensed under the CC BY-SA 3.0[4]
 * [1]: https://stackoverflow.com/questions/261963/how-can-i-iterate-over-an-enum
 * [2]: https://stackoverflow.com/a/31836401
 * [3]: https://stackoverflow.com/users/2938538/francesco-chemolli
 * [4]: http://creativecommons.org/licenses/by-sa/3.0/
 */

ishmemi_type_t bitwise_reduction_types[] = {UCHAR,  USHORT, UINT,   ULONG, ULONGLONG,
                                            INT8,   INT16,  INT32,  INT64, UINT8,
                                            UINT16, UINT32, UINT64, SIZE,  ISHMEMI_TYPE_END};
ishmemi_type_t compare_reduction_types[] = {
    CHAR,   SCHAR,  SHORT,  INT,       LONG, LONGLONG, PTRDIFF, UCHAR,
    USHORT, UINT,   ULONG,  ULONGLONG, INT8, INT16,    INT32,   INT64,
    UINT8,  UINT16, UINT32, UINT64,    SIZE, FLOAT,    DOUBLE,  ISHMEMI_TYPE_END};

ishmemi_type_t arithmetic_reduction_types[] = {
    CHAR,   SCHAR,  SHORT,  INT,       LONG, LONGLONG, PTRDIFF, UCHAR,
    USHORT, UINT,   ULONG,  ULONGLONG, INT8, INT16,    INT32,   INT64,
    UINT8,  UINT16, UINT32, UINT64,    SIZE, FLOAT,    DOUBLE,  ISHMEMI_TYPE_END};

ishmemi_type_t bitwise_amo_types[] = {UINT,  ULONG,  ULONGLONG, INT32,
                                      INT64, UINT32, UINT64,    ISHMEMI_TYPE_END};
ishmemi_type_t standard_amo_types[] = {INT,       LONG,    LONGLONG,        UINT,   ULONG,
                                       ULONGLONG, INT32,   INT64,           UINT32, UINT64,
                                       SIZE,      PTRDIFF, ISHMEMI_TYPE_END};
ishmemi_type_t extended_amo_types[] = {FLOAT,  DOUBLE, INT,       LONG,    LONGLONG,
                                       UINT,   ULONG,  ULONGLONG, INT32,   INT64,
                                       UINT32, UINT64, SIZE,      PTRDIFF, ISHMEMI_TYPE_END};
ishmemi_type_t collectives_copy_types[] = {FLOAT,  DOUBLE, UINT8, UINT16,
                                           UINT32, UINT64, MEM,   ISHMEMI_TYPE_END};
ishmemi_type_t rma_copy_types[] = {FLOAT, DOUBLE, UINT8,  UINT16, UINT32,  UINT64,          MEM,
                                   SIZE8, SIZE16, SIZE32, SIZE64, SIZE128, ISHMEMI_TYPE_END};
ishmemi_type_t strided_rma_copy_types[] = {FLOAT,  DOUBLE, UINT8,   UINT16,
                                           UINT32, UINT64, SIZE8,   SIZE16,
                                           SIZE32, SIZE64, SIZE128, ISHMEMI_TYPE_END};

ishmemi_op_t bitwise_amo_ops[] = {AMO_FETCH_AND, AMO_AND, AMO_FETCH_OR,  AMO_OR,
                                  AMO_FETCH_XOR, AMO_XOR, ISHMEMI_OP_END};
ishmemi_op_t standard_amo_ops[] = {AMO_COMPARE_SWAP, AMO_FETCH_INC, AMO_INC,
                                   AMO_FETCH_ADD,    AMO_ADD,       ISHMEMI_OP_END};
ishmemi_op_t extended_amo_ops[] = {AMO_FETCH, AMO_SET, AMO_SWAP, ISHMEMI_OP_END};

template <typename C, C beginVal, C endVal>
class Iterator {
    typedef typename std::underlying_type<C>::type val_t;
    unsigned int val;

  public:
    Iterator(const C &f) : val(static_cast<val_t>(f)) {}
    Iterator() : val(static_cast<val_t>(beginVal)) {}
    Iterator operator++()
    {
        ++val;
        return *this;
    }
    C operator*()
    {
        return static_cast<C>(val);
    }
    Iterator begin()
    {
        return *this;
    }  // default ctor is good
    Iterator end()
    {
        static const Iterator endIter = ++Iterator(endVal);  // cache it
        return endIter;
    }
    bool operator!=(const Iterator &i)
    {
        return val != i.val;
    }
};

/* test modes */
/*
 * host_host_host - host initiated, destination in host memory, source in host memory
 * host_device_host - host initiated, destination in host memory, source in device memory
 * host_host_device - host initiated, destination in device memory, source in host memory
 * host_device_device - host initiated, destination in device memory, source in device memory
 * on_queue - host initiated, submits device kernel, destination in device memory, source in device
 * memory device - single thread, device initiated, destination and source in device memory
 * device_subgroup - collective device call from a SYCL subgroup
 * device_grp1 - collective call from a 1D SYCL nd_range
 * device_grp2 - collective call from a 2D SYCL nd_range
 * device_grp3 - collective call from a 3D SYCL nd_range
 * device_multi_wg - device_grp1 calls from multiple work groups, dividing the work
 */
typedef enum MODE {
    host_host_host,
    host_host_device,
    host_device_host,
    host_device_device,
    on_queue,
    device,
    device_subgroup,
    device_grp1,
    device_grp2,
    device_grp3,
    device_multi_wg,
    TESTMODE_END
} testmode_t;

typedef Iterator<testmode_t, testmode_t::host_host_host, testmode_t::device_multi_wg>
    testmode_t_Iterator;

typedef Iterator<ishmemi_type_t, ishmemi_type_t::MEM, ishmemi_type_t::SIZE128>
    ishmemi_type_t_Iterator;

typedef Iterator<ishmemi_op_t, ishmemi_op_t::PUT, ishmemi_op_t::DEBUG_TEST> ishmemi_op_t_Iterator;

#define cmd_idle  0
#define cmd_run   1
#define cmd_print 2
#define cmd_exit  3

// PE 0 tells other PEs what to run
struct CMD {
    long cmd;
    size_t iter;
    size_t groups;
    size_t threads;
    size_t nelems;
    ishmemi_type_t type;
    ishmemi_op_t op;
    testmode_t mode;
};

sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

typedef void (*ishmem_test_fn_t)(sycl::queue q, ishmem_team_t *p_wg_teams, int *res, void *dest,
                                 void *src, size_t nelems);

class ishmem_tester {
  protected:  // intended for use by subclasses
    double tsc_frequency;
    char *testname;                 /* from argv[0] */
    int enable_ipc;                 /* from ishmemi_params.ENABLE_GPU_IPC */
    size_t buffer_size;             /* passed in from user */
    long *aligned_source = nullptr; /* source pattern, 64 bit aligned */
    long *aligned_dest = nullptr;   /* destination pattern, 64 bit aligned */
    long *host_source = nullptr;    /* operation source, if in host memory */
    long *host_dest = nullptr;      /* operation destination, if in host memory */
    long *host_result = nullptr;    /* host copy of operation actual results */
    long *host_check = nullptr;     /* host copy of expected results */
    void *device_source = nullptr;  /* source buffer, if in device memory */
    void *device_dest = nullptr;    /* destination buffer, if in device memory */
    int *test_return =
        nullptr;        /* storage to record return value from test funtion, in host memory */
    struct CMD *cmd;    /* used to coordinate PEs */
    struct CMD *devcmd; /* a copy in the device heap */
    bool use_runtime_collectives = true;
    long int *psync; /* used to synchronize PEs */
    bool has_on_queue_implementation = false;
    testmode_t test_modes[(int) TESTMODE_END];
    int num_test_modes = 0;
    ishmemi_type_t test_types[(int) ISHMEMI_TYPE_END];
    int num_test_types = 0;
    ishmemi_op_t test_ops[(int) ISHMEMI_OP_END];
    int num_test_ops = 0;
    uint8_t guard[4096];

    template <typename T>
    size_t tcheck(T *expected, T *actual, size_t nelems);
    bool check_guard(void *p);

    bool source_is_device(testmode_t mode);
    bool dest_is_device(testmode_t mode);
    void print_bw_header();
    void print_bw_result(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t nelems,
                         size_t groups, size_t threads, int pe, double lat_us, double bw_mb);
    /* check results.  The default version uses the pattern created by create_check_pattern */
    virtual size_t check(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t nelems);
    /* return byte size of the source pattern */
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    /* return byte size of the check pattern */
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
    void parse_tester_args(int argc, char *argv[]);  // read command line
    bool parse_test_modes(char *arg);  // build table of modes to test, reports success
    bool parse_test_types(char *arg);  // build table of types to test, reports success
    bool parse_test_ops(char *arg);    // build table of ops to test, reports success
  public:                              // intended for use by users
    int my_pe;                         // set by job launch
    int n_pes;                         // set by job launch
    size_t max_nelems = 1L << 16;      // default value, override on command line
    size_t work_group_size = 1024;     // max and default value, override on command line
    size_t max_groups =
        4;  // max and default value, override on command line, applies to device_multi_wg only
    ishmem_team_t global_wg_teams[max_wg];  //
    int patterndebugflag = 0;               // print patterns, set with command line --patterndebug
    int verboseflag = 0;                    // verbose, set with command line --verbose
    int csvflag = 0;                        // csv mode, set with command line --csv
    bool test_modes_set = false;
    bool test_types_set = false;
    bool test_ops_set = false;

    bool add_test_mode(testmode_t mode);
    bool add_test_type(ishmemi_type_t t);
    bool add_test_op(ishmemi_op_t op);
    bool add_test_mode_list(testmode_t *modes);
    bool add_test_type_list(ishmemi_type_t *ts);
    bool add_test_op_list(ishmemi_op_t *ops);
    void reset_test_modes();
    void reset_test_types();
    void reset_test_ops();

    /* filled in with objects having overloaded operators to call from test kernels */
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_host;
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_on_queue;
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_single;
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_grp1;
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_grp2;
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_grp3;
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_subgroup;
    std::map<std::pair<ishmemi_type_t, ishmemi_op_t>, ishmem_test_fn_t> test_map_fns_multi_wg;

    sycl::queue q;

    ishmem_tester(int argc, char *argv[], bool has_on_queue_impl = false)
        : has_on_queue_implementation{has_on_queue_impl}, q(prop_list)
    {
        ishmem_init();
        validate_runtime();

        for (size_t i = 0; i < 4096; i++) {
            guard[i] = (uint8_t) ((i + 1) % 256);
        }
        for (size_t i = 0; i < ISHMEMI_TYPE_END; i++) {
            test_types[i] = ISHMEMI_TYPE_END;
        }
        for (size_t i = 0; i < ISHMEMI_OP_END; i++) {
            test_ops[i] = ISHMEMI_OP_END;
        }

        testname = basename(argv[0]);
        for (size_t i = 0; i < ISHMEMI_TYPE_END; i++) {
            test_types[i] = ISHMEMI_TYPE_END;
        }
        for (size_t i = 0; i < ISHMEMI_OP_END; i++) {
            test_ops[i] = ISHMEMI_OP_END;
        }
        parse_tester_args(argc, argv);
        /* If there was no testmode list on the command line, use a subset */
        if (num_test_modes == 0) {
            test_modes[num_test_modes++] = device;
            test_modes[num_test_modes++] = device_grp1;
        }

        my_pe = ishmem_my_pe();  // global so called functions can use it
        n_pes = ishmem_n_pes();
        enable_ipc = true;
        const char *ipc_env_val = getenv("ISHMEM_ENABLE_GPU_IPC");
        if (ipc_env_val != nullptr) {
            if (strcasecmp("false", ipc_env_val) == 0) enable_ipc = false;
            else if (strcasecmp("0", ipc_env_val) == 0) enable_ipc = false;
        }
        tsc_frequency = measure_tsc_frequency();
        cmd = (struct CMD *) ishmemi_test_runtime->calloc(2L, sizeof(struct CMD));
        assert(cmd != nullptr);
        devcmd = (struct CMD *) ishmem_calloc(2L, sizeof(struct CMD));
        assert(devcmd != nullptr);
        psync = (long int *) ishmemi_test_runtime->calloc((size_t) n_pes, sizeof(long int));
        assert(psync != nullptr);
        setbuf(stdout, NULL); /* turn off buffering */
        setbuf(stderr, NULL);
        ishmem_sync_all();
        /* Create clones of TEAM_WORLD for use by device_multi_wg tests */
        for (size_t wg = 0; wg < max_groups; wg += 1) {
            int res __attribute__((unused)) = ishmem_team_split_strided(
                ISHMEM_TEAM_WORLD, 0, 1, n_pes, NULL, 0, &global_wg_teams[wg]);
            assert(res == 0);
        }
        printf("[%d] pe %d of %d %d\n", my_pe, my_pe, n_pes, getpid());
    }
    ~ishmem_tester();

    /* alloc_memory is called once the bufsize is computed by main program */
    void alloc_memory(size_t bufsize);

    int finalize_and_report(size_t errors); /* used to return from main() */

    /* the do_test functions are called by the run_ functions, but could be called directly by user
     */
    size_t do_test(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t nelems,
                   unsigned long source_offset, unsigned long dest_offset);
    double do_test_bw(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t groups,
                      size_t threads, size_t iterations, size_t nelems);
    double do_test_bw(testmode_t mode, size_t threads, size_t iterations, size_t nelems);
    size_t run_aligned_tests(
        ishmemi_op_t op);        // run do_test for nelems == 1 up to max_nelems, by doubling
    size_t run_aligned_tests();  // default to op NOP
    size_t run_offset_tests(ishmemi_op_t op);  // run do_test for nelems = 1 to 15 with type-sized
                                               // offsets up to 16 for source and dest
    size_t run_offset_tests();                 // default to op NOP

    /* bw tests do not generate or check patterns, and are called in a loop so that they run for
     * long enough to measure */
    /* bandwidth multiplier is for how many blocks of size typesize() * nelems are transferred per
     * iteration for RMA this is 1, for broadcast it is n_pes, for alltoall it is n_pes^2 this is
     * used only for reporting collective should be false for one-sided ops like amo or rma, and
     * true for collectives
     */
    void run_bw_tests(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, long bandwidth_multiplier,
                      bool collective);
    /* this version runs through the mode, type, and op list set by the -t,y,o switches */
    void run_bw_tests(long bandwidth_multiplier, bool collective);

    /* decode the enum values into strings for printfs */
    const char *mode_to_str(testmode_t mode);
    const char *mode_to_desc(testmode_t mode);
    const char *type_to_str(ishmemi_type_t t);
    const char *op_to_str(ishmemi_op_t op);
    /* returns sizeof() the relevant datatype */
    size_t typesize(ishmemi_type_t t) noexcept;
    /* coordination */
    void sync_all();
    void broadcastcmd();
};

/* forward declarations for functions to be defined by user.  Typically these are defined by calls
 * on the macros below */

SYCL_EXTERNAL int do_test_single(ishmemi_type_t t, ishmemi_op_t op, void *dest, const void *src,
                                 size_t nelems);
template <typename Group>
SYCL_EXTERNAL int do_test_work_group(ishmemi_type_t t, ishmemi_op_t op, void *dest, const void *src,
                                     size_t nelems, const Group &grp);

/* These macros define functions which dispatch on an ishmemi_type_t and call a function
 * specialized to the given type
 *
 * First, define the macro ISHMEM_TYPE_BRANCH, which will contain the code for each arm
 * of the switch statement.  It must end with break; and should start with res = to return a
 * value
 *
 * the arguments of ISHMEM_TYPE_BRANCH are the enum string, the shmem type name,and the C type
 * name.
 *
 * Then instantiate the macto ISHMEM_GEN_TYPE_FUNCTION,.  It requires three arguments: the
 * function signature, the declaration of the return type, and code for the MEM case
 *
 * for example,
 *
 * #define ISHMEM_TYPE_BRANCH(enumname, name, type) res = sizeof(type); break;
 * would work for a function to return the size of each ishmemi_type_t
 *
 * If any arguments of a macro need to contain a "," you must use COMMA instead, otherwise the C
 * preprocessor will interpret the , as a macro argument separator.
 *
 * continuing the sizeof example:
 * ISHMEM_GEN_TYPE_FUNCTION(size_t typesize(ishmemi_type_t t), size_t res = 0;, res = 1; break;)
 * which will declare a function "typesize" which takes an ishmemi_type_t as argument and rturns
 * the size of that type the second argument declares the return type as a size_t, and the third
 * argument supplies the value for the MEM case
 */

// COMMA is needed so you can have , inside the argument of a macro
#define COMMA ,

// #define ISHMEM_TYPE_BRANCH(enum, name, type) break;

#define ISHMEM_GEN_TYPE_FUNCTION(function, returnvar, memcase)                                     \
    function                                                                                       \
    {                                                                                              \
        returnvar switch (t)                                                                       \
        {                                                                                          \
            case MEM:                                                                              \
                memcase;                                                                           \
            case UINT8:                                                                            \
                ISHMEM_TYPE_BRANCH(UINT8, uint8, uint8_t)                                          \
            case UINT16:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT16, uint16, uint16_t)                                       \
            case UINT32:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT32, uint32, uint32_t)                                       \
            case UINT64:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT64, uint64, uint64_t)                                       \
            case ULONGLONG:                                                                        \
                ISHMEM_TYPE_BRANCH(ULONGLONG, ulonglong, unsigned long long)                       \
            case INT8:                                                                             \
                ISHMEM_TYPE_BRANCH(INT8, int8, int8_t)                                             \
            case INT16:                                                                            \
                ISHMEM_TYPE_BRANCH(INT16, int16, int16_t)                                          \
            case INT32:                                                                            \
                ISHMEM_TYPE_BRANCH(INT32, int32, int32_t)                                          \
            case INT64:                                                                            \
                ISHMEM_TYPE_BRANCH(INT64, int64, int64_t)                                          \
            case LONGLONG:                                                                         \
                ISHMEM_TYPE_BRANCH(LONGLONG, longlong, long long)                                  \
            case FLOAT:                                                                            \
                ISHMEM_TYPE_BRANCH(FLOAT, float, float)                                            \
            case DOUBLE:                                                                           \
                ISHMEM_TYPE_BRANCH(DOUBLE, double, double)                                         \
            case LONGDOUBLE:                                                                       \
                ISHMEM_TYPE_BRANCH(LONG, long, long)                                               \
            case CHAR:                                                                             \
                ISHMEM_TYPE_BRANCH(CHAR, char, char)                                               \
            case SCHAR:                                                                            \
                ISHMEM_TYPE_BRANCH(SCHAR, schar, signed char)                                      \
            case SHORT:                                                                            \
                ISHMEM_TYPE_BRANCH(SHORT, short, short)                                            \
            case INT:                                                                              \
                ISHMEM_TYPE_BRANCH(INT, int, int)                                                  \
            case LONG:                                                                             \
                ISHMEM_TYPE_BRANCH(LONG, long, long)                                               \
            case UCHAR:                                                                            \
                ISHMEM_TYPE_BRANCH(UCHAR, uchar, unsigned char)                                    \
            case USHORT:                                                                           \
                ISHMEM_TYPE_BRANCH(USHORT, ushort, unsigned short)                                 \
            case UINT:                                                                             \
                ISHMEM_TYPE_BRANCH(UINT, uint, unsigned int)                                       \
            case ULONG:                                                                            \
                ISHMEM_TYPE_BRANCH(ULONG, ulong, unsigned long)                                    \
            case SIZE:                                                                             \
                ISHMEM_TYPE_BRANCH(SIZE, size, size_t)                                             \
            case PTRDIFF:                                                                          \
                ISHMEM_TYPE_BRANCH(PTRDIFF, ptrdiff, ptrdiff_t);                                   \
            case SIZE8:                                                                            \
                ISHMEM_TYPE_BRANCH(SIZE8, 8, uint8_t)                                              \
            case SIZE16:                                                                           \
                ISHMEM_TYPE_BRANCH(SIZE16, 16, uint16_t)                                           \
            case SIZE32:                                                                           \
                ISHMEM_TYPE_BRANCH(SIZE32, 32, uint32_t)                                           \
            case SIZE64:                                                                           \
                ISHMEM_TYPE_BRANCH(SIZE64, 64, uint64_t)                                           \
            case SIZE128:                                                                          \
                ISHMEM_TYPE_BRANCH(SIZE128, 128, __uint128_t)                                      \
            default:                                                                               \
                assert(0);                                                                         \
        }                                                                                          \
        return (res);                                                                              \
    }

#define ISHMEM_GEN_MODE_FUNCTION(function)                                                         \
    function                                                                                       \
    {                                                                                              \
        const char *res;                                                                           \
        switch (mode) {                                                                            \
            case host_host_host:                                                                   \
                ISHMEM_MODE_BRANCH(host_host_host)                                                 \
            case host_host_device:                                                                 \
                ISHMEM_MODE_BRANCH(host_host_device)                                               \
            case host_device_host:                                                                 \
                ISHMEM_MODE_BRANCH(host_device_host)                                               \
            case host_device_device:                                                               \
                ISHMEM_MODE_BRANCH(host_device_device)                                             \
            case on_queue:                                                                         \
                ISHMEM_MODE_BRANCH(on_queue)                                                       \
            case device:                                                                           \
                ISHMEM_MODE_BRANCH(device)                                                         \
            case device_subgroup:                                                                  \
                ISHMEM_MODE_BRANCH(device_subgroup)                                                \
            case device_grp1:                                                                      \
                ISHMEM_MODE_BRANCH(device_grp1)                                                    \
            case device_grp2:                                                                      \
                ISHMEM_MODE_BRANCH(device_grp2)                                                    \
            case device_grp3:                                                                      \
                ISHMEM_MODE_BRANCH(device_grp3)                                                    \
            case device_multi_wg:                                                                  \
                ISHMEM_MODE_BRANCH(device_multi_wg)                                                \
            default:                                                                               \
                res = "unknown";                                                                   \
        }                                                                                          \
        return (res);                                                                              \
    }

#define ISHMEM_GEN_TEST_FUNCTION_SINGLE(returnvar, memcase)                                        \
    ISHMEM_GEN_TYPE_FUNCTION(                                                                      \
        int do_test_single(ishmemi_type_t t COMMA ishmemi_op_t op COMMA void *dest                 \
                               COMMA const void *src COMMA size_t nelems),                         \
        returnvar, memcase)

#define ISHMEM_GEN_TEST_FUNCTION_WORK_GROUP(returnvar, memcase)                                    \
    ISHMEM_GEN_TYPE_FUNCTION(                                                                      \
        template int do_test_work_group<sycl::group<1>>(                                           \
            ishmemi_type_t t COMMA ishmemi_op_t op COMMA void *dest COMMA const void *src COMMA    \
                size_t nelems COMMA const sycl::group<1> &grp);                                    \
        template int do_test_work_group<sycl::group<2>>(                                           \
            ishmemi_type_t t COMMA ishmemi_op_t op COMMA void *dest COMMA const void *src COMMA    \
                size_t nelems COMMA const sycl::group<2> &grp);                                    \
        template int do_test_work_group<sycl::group<3>>(                                           \
            ishmemi_type_t t COMMA ishmemi_op_t op COMMA void *dest COMMA const void *src COMMA    \
                size_t nelems COMMA const sycl::group<3> &grp);                                    \
        template int do_test_work_group<sycl::sub_group>(                                          \
            ishmemi_type_t t COMMA ishmemi_op_t op COMMA void *dest COMMA const void *src COMMA    \
                size_t nelems COMMA const sycl::sub_group &grp);                                   \
        template <typename Group> SYCL_EXTERNAL int do_test_work_group(                            \
            ishmemi_type_t t COMMA ishmemi_op_t op COMMA void *dest COMMA const void *src COMMA    \
                size_t nelems COMMA const Group &grp),                                             \
        returnvar, memcase)

template <typename Group>
SYCL_EXTERNAL int do_test_work_group(ishmemi_type_t t, ishmemi_op_t op, void *dest, const void *src,
                                     size_t nelems, const Group &grp);

/* these are here to keep the compiler happy, if needed */
#ifndef BW_TEST_HEADER
#define BW_TEST_HEADER
#endif
#ifndef BW_TEST_FUNCTION
#define BW_TEST_FUNCTION
#endif
#ifndef BW_TEST_FUNCTION_ON_QUEUE
#define BW_TEST_FUNCTION_ON_QUEUE
#endif
#ifndef BW_TEST_FUNCTION_WORK_GROUP
#define BW_TEST_FUNCTION_WORK_GROUP
#endif

/* this defines a function
const char *ishmem_tester::strtomode(testmode_t mode);
which returns the shmem name corresponding to the enum type passed in
*/

#ifdef ISHMEM_MODE_BRANCH
#undef ISHMEM_MODE_BRANCH
#endif

#define ISHMEM_MODE_BRANCH(name)                                                                   \
    res = #name;                                                                                   \
    break;

ISHMEM_GEN_MODE_FUNCTION(const char *ishmem_tester::mode_to_str(testmode_t mode))

/* This defines a function
size_t ishmem_tester::typesize(ishmemi_type_t t);
which returns the sizeof(the type corresponding to the enum passed in
*/
size_t ishmem_tester::typesize(ishmemi_type_t t) noexcept
{
    size_t res = 0;
    switch (t) {
        case NONE:
            break;
        case FLOAT:
            res = sizeof(float);
            break;
        case DOUBLE:
            res = sizeof(double);
            break;
        case LONGDOUBLE:
            res = sizeof(long double);
            break;
        case CHAR:
            res = sizeof(char);
            break;
        case SCHAR:
            res = sizeof(signed char);
            break;
        case SHORT:
            res = sizeof(short);
            break;
        case INT:
            res = sizeof(int);
            break;
        case LONG:
            res = sizeof(long);
            break;
        case LONGLONG:
            res = sizeof(long long);
            break;
        case UCHAR:
            res = sizeof(unsigned char);
            break;
        case USHORT:
            res = sizeof(unsigned short);
            break;
        case UINT:
            res = sizeof(unsigned int);
            break;
        case ULONG:
            res = sizeof(unsigned long);
            break;
        case ULONGLONG:
            res = sizeof(unsigned long long);
            break;
        case INT8:
            res = sizeof(int8_t);
            break;
        case INT16:
            res = sizeof(int16_t);
            break;
        case INT32:
            res = sizeof(int32_t);
            break;
        case INT64:
            res = sizeof(int64_t);
            break;
        case UINT8:
            res = sizeof(uint8_t);
            break;
        case UINT16:
            res = sizeof(uint16_t);
            break;
        case UINT32:
            res = sizeof(uint32_t);
            break;
        case UINT64:
            res = sizeof(uint64_t);
            break;
        case SIZE:
            res = sizeof(size_t);
            break;
        case PTRDIFF:
            res = sizeof(ptrdiff_t);
            break;
        case SIZE8:
            res = sizeof(uint8_t);
            break;
        case SIZE16:
            res = sizeof(uint16_t);
            break;
        case SIZE32:
            res = sizeof(uint32_t);
            break;
        case SIZE64:
            res = sizeof(uint64_t);
            break;
        case SIZE128:
            res = sizeof(__uint128_t);
            break;
        case MEM:
            res = 1;
            break;
        case ISHMEMI_TYPE_END:
            res = 0;
    }
    return (res);
}

template <typename T>
size_t ishmem_tester::tcheck(T *expected, T *actual, size_t nelems)
{
    size_t errors = 0;
    for (size_t idx = 0; idx < nelems; idx += 1) {
        if constexpr (std::is_same_v<T, float>) {
            T got = actual[idx];
            T expval = expected[idx];
            bool thiserr;
            if (expval == 0.0) thiserr = std::fabs(got - expval) > 0.00001;
            else thiserr = std::fabs((got - expval) / expval) > 0.00001;
            if (thiserr) {
                errors += 1;
                if (errors <= 16) {
                    printf("[%d] err idx 0x%x nelems %ld got %f (%8x) expected %f (%8x)\n", my_pe,
                           (unsigned int) idx, nelems, got, *((uint32_t *) &actual[idx]), expval,
                           *((uint32_t *) &expected[idx]));
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            T got = actual[idx];
            T expval = expected[idx];
            bool thiserr;
            if (expval == 0.0) thiserr = std::fabs(got - expval) > 0.000000001;
            else thiserr = std::fabs((got - expval) / expval) > 0.000000001;
            if (thiserr) {
                errors += 1;
                if (errors <= 16) {
                    printf("[%d] err idx 0x%x nelems %ld got %f (%16lx) expected %f (%16lx)\n",
                           my_pe, (unsigned int) idx, nelems, got, *((uint64_t *) &actual[idx]),
                           expval, *((uint64_t *) &expected[idx]));
                }
            }
        } else {
            T got = actual[idx];
            T expval = expected[idx];

            if (got != expval) {
                errors += 1;
                if (errors <= 16) {
                    printf("[%d] err idx 0x%x nelems %ld got %016lx expected %016lx\n", my_pe,
                           (unsigned int) idx, nelems, (long) got, (long) expval);
                }
            }
        }
    }
    return (errors);
}

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = tcheck((type *) host_check, (type *) host_result, nelems);                               \
    break;

ISHMEM_GEN_TYPE_FUNCTION(size_t ishmem_tester::check(ishmemi_type_t t COMMA ishmemi_op_t op COMMA
                                                         testmode_t mode COMMA size_t nelems),
                         size_t res = 0;
                         , tcheck((uint8_t *) host_check COMMA(uint8_t *) host_result COMMA nelems);
                         break;)

static void print_tester_usage()
{
    std::cerr << "Usage: \n";
    std::cerr << "  <launcher> -n 2 [launcher-options] ./scripts/ishmrun <test> [test-options]\n";
    std::cerr << "test-options: \n";
    std::cerr << "  --max_nelems, -m    Set the max message size that will be"
                 " operated on\n"
                 "  --max_groups, -g  Set the max number of groups for mode device_multi_wg"
                 "  --work_group_size, -w  Set the dimensions of the device kernel's"
                 " entire index space (only used in multithreaded tests)\n"
                 " before measurements begin\n"
                 "  --csv -c          Output in csv format\n"
                 "  --patterndebug -p Print testpatterns\n"
                 "  --test_modes -t mode[,mode]* | all  Select modes to test\n"
                 "  --verbose -v      Print each test\n"
                 "  --help,  -h       Print usage message\n";
    exit(1);
}

static bool tester_isPowerOfTwo(unsigned long n)
{
    return (__builtin_popcountl(n) == 1);
}

bool ishmem_tester::add_test_mode(testmode_t mode)
{
    if (mode == testmode_t::on_queue && !has_on_queue_implementation) {
        std::cerr << "Error: There is no \"on_queue\" variation of the function under test."
                  << std::endl;
        exit(1);
    }

    if (num_test_modes >= TESTMODE_END) return false;
    test_modes[num_test_modes++] = mode;
    test_modes_set = true;
    return (true);
}

bool ishmem_tester::add_test_mode_list(testmode_t *modes)
{
    while (*modes != TESTMODE_END)
        if (!add_test_mode(*modes++)) return (false);
    return (true);
}

bool ishmem_tester::add_test_type(ishmemi_type_t t)
{
    if (num_test_types >= ISHMEMI_TYPE_END) return false;
    test_types[num_test_types++] = t;
    test_types_set = true;
    return (true);
}

bool ishmem_tester::add_test_type_list(ishmemi_type_t *ts)
{
    while (*ts != ISHMEMI_TYPE_END)
        if (!add_test_type(*ts++)) return (false);
    return (true);
}

bool ishmem_tester::add_test_op(ishmemi_op_t op)
{
    if (num_test_ops >= ISHMEMI_OP_END) return false;
    test_ops[num_test_ops++] = op;
    test_ops_set = true;
    return (true);
}

bool ishmem_tester::add_test_op_list(ishmemi_op_t *ops)
{
    while (*ops != ISHMEMI_OP_END)
        if (!add_test_op(*ops++)) return (false);
    return (true);
}

void ishmem_tester::reset_test_modes()
{
    num_test_modes = 0;
}

void ishmem_tester::reset_test_types()
{
    num_test_types = 0;
}

void ishmem_tester::reset_test_ops()
{
    num_test_ops = 0;
}

/* input is a string like mode,mode,mode  or "all" */
bool ishmem_tester::parse_test_modes(char *arg)
{
    test_modes_set = true;
    if (strcmp("all", arg) == 0) {
        num_test_modes = 0;
        for (testmode_t mode : testmode_t_Iterator()) {
            if (!(mode == testmode_t::on_queue && !has_on_queue_implementation)) {
                add_test_mode(mode);
            }
        }
        return true;
    }
    char *saveptr;
    char *m = strtok_r(arg, ",", &saveptr);
    while (m) {
        bool found = false;
        for (testmode_t mode : testmode_t_Iterator()) {
            if (strcmp(mode_to_str(mode), m) == 0) {
                if (!add_test_mode(mode)) {
                    printf("too many test modes\n");
                    return (false);
                }
                found = true;
                break;
            }
        }
        if (!found) {
            printf("unknown test mode %s\n", m);
            return (false);
        }
        m = strtok_r(nullptr, ",", &saveptr);
    }
    return (true);
}

/* input is a string like mode,mode,mode  or "all" */
bool ishmem_tester::parse_test_types(char *arg)
{
    test_types_set = true;
    if (strcmp("all", arg) == 0) {
        add_test_type(UINT8);
        add_test_type(UINT16);
        add_test_type(UINT32);
        add_test_type(UINT64);
        add_test_type(INT8);
        add_test_type(INT16);
        add_test_type(INT32);
        add_test_type(INT64);
        add_test_type(FLOAT);
        add_test_type(DOUBLE);
        return true;
    }
    char *saveptr;
    char *m = strtok_r(arg, ",", &saveptr);
    while (m) {
        bool found = false;
        for (ishmemi_type_t t : ishmemi_type_t_Iterator()) {
            const char *key = type_to_str(t);
            if (key == NULL) break;
            if (strcmp(key, m) == 0) {
                if (!add_test_type(t)) {
                    printf("too many test types\n");
                    return (false);
                }
                found = true;
                break;
            }
        }
        if (!found) {
            printf("unknown test type %s\n", m);
            return (false);
        }
        m = strtok_r(nullptr, ",", &saveptr);
    }
    return (true);
}

/* input is a string like mode,mode,mode  or "all" */
bool ishmem_tester::parse_test_ops(char *arg)
{
    test_ops_set = true;
    if (strcmp("reduce", arg) == 0) {
        add_test_op(AND_REDUCE);
        add_test_op(OR_REDUCE);
        add_test_op(XOR_REDUCE);
        add_test_op(MIN_REDUCE);
        add_test_op(MAX_REDUCE);
        add_test_op(SUM_REDUCE);
        add_test_op(PROD_REDUCE);
        return true;
    }
    if (strcmp("amo", arg) == 0) {
        add_test_op(AMO_FETCH);
        add_test_op(AMO_SET);
        add_test_op(AMO_COMPARE_SWAP);
        add_test_op(AMO_SWAP);
        add_test_op(AMO_FETCH_INC);
        add_test_op(AMO_INC);
        add_test_op(AMO_FETCH_ADD);
        add_test_op(AMO_ADD);
        add_test_op(AMO_FETCH_AND);
        add_test_op(AMO_AND);
        add_test_op(AMO_FETCH_OR);
        add_test_op(AMO_OR);
        add_test_op(AMO_FETCH_XOR);
        add_test_op(AMO_XOR);
        return true;
    }
    char *saveptr;
    char *m = strtok_r(arg, ",", &saveptr);
    while (m) {
        bool found = false;
        for (ishmemi_op_t op : ishmemi_op_t_Iterator()) {
            if (strcmp(op_to_str(op), m) == 0) {
                if (!add_test_op(op)) {
                    printf("too many test modes\n");
                    return (false);
                }
                found = true;
                break;
            }
        }
        if (!found) {
            printf("unknown test type %s\n", m);
            return (false);
        }
        m = strtok_r(nullptr, ",", &saveptr);
    }
    return (true);
}

void ishmem_tester::parse_tester_args(int argc, char *argv[])
{
    static struct option long_opts[] = {{"max_nelems", required_argument, nullptr, 'm'},
                                        {"max_groups", required_argument, nullptr, 'g'},
                                        {"work_group_size", required_argument, nullptr, 'w'},
                                        {"patterndebug", no_argument, &patterndebugflag, 'p'},
                                        {"verbose", no_argument, &verboseflag, 'v'},
                                        {"csv", no_argument, &csvflag, 'c'},
                                        {"help", required_argument, nullptr, 'h'},
                                        {"test_modes", required_argument, nullptr, 't'},
                                        {"test_types", required_argument, nullptr, 'y'},
                                        {"test_ops", required_argument, nullptr, 'o'},
                                        {"no_shmem_collectives", no_argument, nullptr, 'n'},
                                        {0, 0, nullptr, 0}};
    while (true) {
        const auto opt = getopt_long(argc, argv, "m:g:w:pvch:t:y:o:n", long_opts, nullptr);
        if (opt == -1) break;
        unsigned long val;
        if (optarg) val = (unsigned long) strtol(optarg, NULL, 0);
        else val = 0;
        switch (opt) {
            case 'm':
                if (tester_isPowerOfTwo(val)) max_nelems = val;
                else {
                    std::cerr << "Error: The argument for max nelems must be a "
                                 "positive integer and a power of two"
                              << std::endl;
                    exit(1);
                }
                break;
            case 'g':
                if (val > 0 && val <= max_wg) max_groups = val;
                else {
                    std::cerr << "Error: The argument for the max work group size must be "
                                 "a positive integer that is no greater than 256"
                              << std::endl;
                }
                break;
            case 'w':
                if (val > 0 && val <= 1024) work_group_size = val;
                else {
                    std::cerr << "Error: The argument for the max work group size must be "
                                 "a positive integer that is no greater than 1024"
                              << std::endl;
                }
                break;
            case 't':
                if (!parse_test_modes(optarg)) {
                    print_tester_usage();
                    exit(1);
                }
                break;
            case 'y':
                if (!parse_test_types(optarg)) {
                    print_tester_usage();
                    exit(1);
                }
                break;
            case 'o':
                if (!parse_test_ops(optarg)) {
                    print_tester_usage();
                    exit(1);
                }
                break;
            case 'n':
                use_runtime_collectives = false;
                break;
            case 'h':
            case '?':
                print_tester_usage();
                exit(1);
            default:
                break;
        }
    }
}

#define check_alloc(x)                                                                             \
    {                                                                                              \
        ISHMEM_DEBUG_MSG("[%d] %s=%p\n", my_pe, #x, x);                                            \
        assert(x != nullptr);                                                                      \
        memcpy((void *) (((uintptr_t) x) + buffer_size), guard, 4096);                             \
    }

void ishmem_tester::alloc_memory(size_t bufsize)
{
    buffer_size = bufsize;
    bufsize += 4096; /* create a guard band */
    /* allocate host memory */
    aligned_source = (long *) malloc(bufsize); /* data pattern */
    check_alloc(aligned_source);
    aligned_dest = (long *) malloc(bufsize); /* data pattern */
    check_alloc(aligned_dest);

    host_check = (long *) malloc(bufsize); /* expected data */
    check_alloc(host_check);
    host_source = (long *) ishmemi_test_runtime->malloc(bufsize); /* source data for this PE */
    check_alloc(host_source);
    host_dest = (long *) ishmemi_test_runtime->malloc(bufsize); /* source data for this PE */
    check_alloc(host_dest);
    host_result = (long *) malloc(bufsize); /* used to read back actual data */
    check_alloc(host_result);
    /* allocate GPU memory for source and destination */
    /* the extra 1024 is for the larger offsets, shouldn't be needed */
    device_source = (long *) ishmem_malloc(bufsize); /* gpu source, if used */
    //  cannot use guards in gpu memory check_alloc(device_source);
    assert(device_source != nullptr);
    device_dest = (long *) ishmem_malloc(bufsize); /* gpu destination, if used */
    // check_alloc(device_dest);
    assert(device_source != nullptr);

    test_return = sycl::malloc_host<int>(1, q);
    assert(test_return != nullptr);
}

#define check_free(fn, x)                                                                          \
    {                                                                                              \
        ISHMEM_DEBUG_MSG("[%d] %s=%p\n", my_pe, #x, x);                                            \
        assert(x);                                                                                 \
        assert(check_guard(x));                                                                    \
        fn(x);                                                                                     \
        x = nullptr;                                                                               \
    }

ishmem_tester::~ishmem_tester()
{
    check_free(free, aligned_source);
    check_free(free, aligned_dest);
    check_free(free, host_check);
    check_free(free, host_result);
    ishmemi_test_runtime->free(cmd);
    check_free(ishmemi_test_runtime->free, host_source);
    check_free(ishmemi_test_runtime->free, host_dest);
    ishmemi_test_runtime->free(psync);
    ishmem_free(devcmd);
    ishmem_free(device_source);
    ishmem_free(device_dest);
    assert(test_return);
    sycl::free(test_return, q);
    test_return = nullptr;
    ishmem_finalize();
}

void ishmem_tester::sync_all()
{
    if (use_runtime_collectives) {
        ishmemi_test_runtime->sync();
    } else {
        q.single_task([=]() { ishmem_sync_all(); }).wait_and_throw();
    }
}

void ishmem_tester::broadcastcmd()
{
    if (use_runtime_collectives) {
        ishmemi_test_runtime->sync();
        ishmemi_test_runtime->broadcast(&cmd[1], &cmd[0], sizeof(struct CMD), 0);
    } else {
        struct CMD *ldevcmd = devcmd;
        if (my_pe == 0) {
            q.memcpy(&ldevcmd[0], &cmd[0], sizeof(struct CMD)).wait_and_throw();
        }
        q.single_task([=]() {
             ishmem_sync_all();
             ishmem_broadcastmem(&ldevcmd[1], &ldevcmd[0], sizeof(struct CMD), 0);
         }).wait_and_throw();
        q.memcpy(&cmd[1], &ldevcmd[1], sizeof(struct CMD)).wait_and_throw();
    }
}

/* result written into aligned_source, using host_source as a temp buffer */
size_t ishmem_tester::create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                            size_t nelems)
{
    long int from_pe = (long int) my_pe;
    long int to_pe = (long int) (my_pe + 1) % n_pes;
    size_t test_size = nelems * typesize(t);
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        aligned_source[idx] = ((long) nelems << 48) + ((0x80L + from_pe) << 40) +
                              ((0x80L + to_pe) << 32) + (long) idx;
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] source pattern idx %lu val %016lx\n", my_pe, idx, aligned_source[idx]);
        }
    }
    return (nelems * typesize(t));
}

/* check pattern written into host_check, using host_source as a temp buffer */
size_t ishmem_tester::create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                           size_t nelems)
{
    long int from_pe = (long int) (my_pe + n_pes - 1) % n_pes;
    long int to_pe = (long int) my_pe;
    size_t test_size = nelems * typesize(t);
    for (size_t idx = 0; idx < ((test_size / sizeof(long)) + 1); idx += 1) {
        host_check[idx] = ((long) nelems << 48) + ((0x80L + from_pe) << 40) +
                          ((0x80L + to_pe) << 32) + (long) idx;
        if (patterndebugflag && (idx < 16)) {
            printf("[%d] check pattern idx %lu val %016lx\n", my_pe, idx, host_check[idx]);
        }
    }
    return (test_size);
}

bool ishmem_tester::source_is_device(testmode_t mode)
{
    switch (mode) {
        case host_host_host:
        case host_host_device:
            return (false);
        default:
            return (true);
    }
}

bool ishmem_tester::dest_is_device(testmode_t mode)
{
    switch (mode) {
        case host_host_host:
        case host_device_host:
            return (false);
        default:
            return (true);
    }
}

bool ishmem_tester::check_guard(void *p)
{
    return (memcmp((void *) (((uintptr_t) p) + buffer_size), guard, 4096) == 0);
}

size_t ishmem_tester::do_test(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t nelems,
                              unsigned long source_offset, unsigned long dest_offset)
{
    void *test_source = (source_is_device(mode)) ? device_source : host_source;
    void *test_dest = (dest_is_device(mode)) ? device_dest : host_dest;
    if (t == LONGDOUBLE) return (0); /* do not report errors here */

    size_t check_size = create_check_pattern(t, op, mode, nelems);
    memset(aligned_dest, 128 + my_pe, check_size);
    assert(check_size <= buffer_size);
    assert(check_guard(aligned_dest));
    size_t source_size = create_source_pattern(t, op, mode, nelems);
    assert(source_size <= buffer_size);
    assert(check_guard(aligned_source));
    /* at this point, aligned_source is correct, we will copy to device_source if needed */
    test_source = (void *) (((uintptr_t) test_source) + source_offset);
    test_dest = (void *) (((uintptr_t) test_dest) + dest_offset);
    int *local_test_return = test_return; /* local variable needed for sycl lambda capture */
    if (source_is_device(mode)) {
        q.memcpy(test_source, aligned_source, source_size).wait_and_throw();
    } else {
        memcpy(test_source, aligned_source, source_size);
    }
    if (dest_is_device(mode)) {
        q.memcpy(test_dest, aligned_dest, check_size).wait_and_throw(); /* prefill destination */
    } else {
        memcpy(test_dest, aligned_dest, check_size); /* prefill destination */
    }
    ishmem_test_fn_t fnp_host = test_map_fns_host[std::make_pair(t, op)];
    ishmem_test_fn_t fnp_on_queue = test_map_fns_on_queue[std::make_pair(t, op)];
    ishmem_test_fn_t fnp_single = test_map_fns_single[std::make_pair(t, op)];
    ishmem_test_fn_t fnp_grp1 = test_map_fns_grp1[std::make_pair(t, op)];
    ishmem_test_fn_t fnp_grp2 = test_map_fns_grp2[std::make_pair(t, op)];
    ishmem_test_fn_t fnp_grp3 = test_map_fns_grp3[std::make_pair(t, op)];
    ishmem_test_fn_t fnp_subgroup = test_map_fns_subgroup[std::make_pair(t, op)];
    ishmem_test_fn_t fnp_multi_wg = test_map_fns_multi_wg[std::make_pair(t, op)];
    switch (mode) {
        case host_host_host:
        case host_host_device:
        case host_device_host:
        case host_device_device: {
            if (fnp_host != NULL)
                fnp_host(q, &global_wg_teams[0], local_test_return, test_dest, test_source, nelems);
            break;
        }
        case on_queue: {
            if (fnp_on_queue != NULL)
                fnp_on_queue(q, &global_wg_teams[0], local_test_return, test_dest, test_source,
                             nelems);
            break;
        }
        case device: {
            if (fnp_single != NULL)
                fnp_single(q, &global_wg_teams[0], local_test_return, test_dest, test_source,
                           nelems);
            break;
        }
        case device_subgroup: {
            if (fnp_subgroup != NULL)
                fnp_subgroup(q, &global_wg_teams[0], local_test_return, test_dest, test_source,
                             nelems);
            break;
        }
        case device_grp1: {
            if (fnp_grp1 != NULL)
                fnp_grp1(q, &global_wg_teams[0], local_test_return, test_dest, test_source, nelems);
            break;
        }
        case device_grp2: {
            if (fnp_grp2 != NULL)
                fnp_grp2(q, &global_wg_teams[0], local_test_return, test_dest, test_source, nelems);
            break;
        }
        case device_grp3: {
            if (fnp_grp3 != NULL)
                fnp_grp3(q, &global_wg_teams[0], local_test_return, test_dest, test_source, nelems);
            break;
        }
        case device_multi_wg: {
            if (fnp_multi_wg != NULL)
                fnp_multi_wg(q, &global_wg_teams[0], local_test_return, test_dest, test_source,
                             nelems);
            break;
        }
        default:
            assert(0);
            break;
    }
    if (*local_test_return != 0) {
        printf("[%d] Test %s datatype %s op %s nelems %ld FAIL return value %d\n", my_pe,
               mode_to_desc(mode), type_to_str(t), op_to_str(op), nelems, *local_test_return);
    }
    size_t errors = 0;
    if (dest_is_device(mode)) q.memcpy(host_result, test_dest, check_size).wait_and_throw();
    else memcpy(host_result, test_dest, check_size);
    /* check routine.  The default one compares host_result with host_check */
    errors = check(t, op, mode, check_size / typesize(t));
    if (errors > 0) {
        printf("[%d] Test %s datatype %s op %s nelems %ld errors %ld\n", my_pe, mode_to_desc(mode),
               type_to_str(t), op_to_str(op), nelems, errors);
    }

    return (errors);
}

double ishmem_tester::do_test_bw(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t groups,
                                 size_t threads, size_t iterations, size_t nelems)
{
    void *lsrc __attribute__((unused)) = nullptr;
    void *ldest __attribute__((unused)) = nullptr;

    lsrc = (source_is_device(mode)) ? device_source : host_source;
    ldest = (dest_is_device(mode)) ? device_dest : host_dest;
    double duration = 0;
    nelems /= groups;  // will be 1 except for mode device_multi_wg
    size_t groupsize = nelems * typesize(t);
    BW_TEST_HEADER;  // define any needed variables
    switch (mode) {
        case host_host_host:
        case host_host_device:
        case host_device_host:
        case host_device_device: {
            void *src __attribute__((unused)) = lsrc;
            void *dest __attribute__((unused)) = ldest;
            unsigned long start = rdtsc();
            BW_TEST_FUNCTION;
            unsigned long stop = rdtsc();
            duration = ((double) (stop - start)) / tsc_frequency;
            break;
        }
        case on_queue: {
            void *src __attribute__((unused)) = lsrc;
            void *dest __attribute__((unused)) = ldest;
            unsigned long start = rdtsc();
            BW_TEST_FUNCTION_ON_QUEUE;
            q.wait_and_throw();
            ishmem_quiet();
            unsigned long stop = rdtsc();
            duration = ((double) (stop - start)) / tsc_frequency;
            break;
            break;
        }
        case device: {
            void *src __attribute__((unused)) = lsrc;
            void *dest __attribute__((unused)) = ldest;
            auto e = q.single_task([=]() { BW_TEST_FUNCTION; });
            e.wait_and_throw();
            duration = getduration(e);
            break;
        }
        case device_subgroup:
        case device_grp3:
        case device_grp2:
        case device_grp1: {
            void *src __attribute__((unused)) = lsrc;
            void *dest __attribute__((unused)) = ldest;
            auto e = q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) threads),
                                                      sycl::range<1>((size_t) threads)),
                                    [=](sycl::nd_item<1> it) {
                                        auto grp __attribute__((unused)) = it.get_group();
                                        BW_TEST_FUNCTION_WORK_GROUP;
                                    });
            e.wait_and_throw();
            duration = getduration(e);
            break;
        }
        case device_multi_wg: {
            auto e = q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) threads * groups),
                                                      sycl::range<1>((size_t) threads)),
                                    [=](sycl::nd_item<1> it) {
                                        auto grp __attribute__((unused)) = it.get_group();
                                        size_t mygroup = it.get_global_linear_id() /
                                                         threads; /* which wg are we */
                                        void *src __attribute((unused)) =
                                            (void *) (((uintptr_t) lsrc) + (groupsize * mygroup));
                                        void *dest __attribute((unused)) =
                                            (void *) (((uintptr_t) ldest) + (groupsize * mygroup));
                                        BW_TEST_FUNCTION_WORK_GROUP;
                                    });
            e.wait_and_throw();
            duration = getduration(e);
            break;
        }
        default:
            assert(0);
            break;
    }
    return (duration);
}

double ishmem_tester::do_test_bw(testmode_t mode, size_t threads, size_t iterations, size_t nelems)
{
    return (do_test_bw(LONG, NOP, mode, 1L, threads, iterations, nelems));
}

size_t ishmem_tester::run_aligned_tests(ishmemi_op_t op)
{
    size_t errors = 0;
    printf("[%d] Run Aligned Tests op %s\n", my_pe, op_to_str(op));
    for (int i = 0; i < num_test_modes; i += 1) {
        testmode_t mode = test_modes[i];
        printf("[%d] Testing %s\n", my_pe, mode_to_desc(mode));
        /* test all datatypes */
        for (int typeindex = 0; typeindex < num_test_types; typeindex += 1) {
            ishmemi_type_t t = test_types[typeindex];
            /* test power of two sizes */
            for (size_t nelems = 1; nelems <= max_nelems; nelems <<= 1) {
                if (verboseflag && (my_pe == 0)) {
                    printf("[%d] Test %s %s %s nelems %ld os %ld od %ld\n", my_pe,
                           mode_to_desc(mode), type_to_str(t), op_to_str(op), nelems, 0L, 0L);
                }
                errors += do_test(t, op, mode, nelems, 0L, 0L);
                /* selection of buffers is done in do_test */
            }
        }
    }
    return (errors);
}

size_t ishmem_tester::run_aligned_tests()
{
    size_t errors = 0;
    for (int op_index = 0; op_index < num_test_ops; op_index += 1) {
        ishmemi_op_t op = test_ops[op_index];
        errors += run_aligned_tests(op);
    }
    return (errors);
}

size_t ishmem_tester::run_offset_tests(ishmemi_op_t op)
{
    size_t errors = 0;
    /* quick tests of different source and destination offsets and small lengths */
    /* could be sped up by making the numbers of cases datatype dependent */
    printf("[%d] Run Offset Tests op %s\n", my_pe, op_to_str(op));
    for (int i = 0; i < num_test_modes; i += 1) {
        testmode_t mode = test_modes[i];
        printf("[%d] Testing %s\n", my_pe, mode_to_desc(mode));
        for (int typeindex = 0; typeindex < num_test_types; typeindex += 1) {
            ishmemi_type_t t = test_types[typeindex];
            for (size_t nelems = 1; nelems <= 16; nelems += 1) {
                /* offsets run from 0 to 15 in units of the datatype size */
                for (unsigned long source_offset = 0; source_offset < 15;
                     source_offset += typesize(t)) {
                    for (unsigned long dest_offset = 0; dest_offset < 15;
                         dest_offset += typesize(t)) {
                        if (verboseflag && (my_pe == 0)) {
                            printf("[%d] Test %s %s nelems %ld os %ld od %ld\n", my_pe,
                                   mode_to_desc(mode), type_to_str(t), nelems, source_offset,
                                   dest_offset);
                        }
                        errors += do_test(t, op, mode, nelems, source_offset, dest_offset);
                    }
                }
            }
        }
    }
    return (errors);
}

size_t ishmem_tester::run_offset_tests()
{
    size_t errors = 0;
    for (int op_index = 0; op_index < num_test_ops; op_index += 1) {
        ishmemi_op_t op = test_ops[op_index];
        errors += run_offset_tests(op);
    }
    return (errors);
}

void ishmem_tester::print_bw_header()
{
    if (csvflag && (my_pe == 0))
        printf("csv,testname,ipc,npes,type,op,mode,groups,threads,bytes,pe,latency_us,bw_mb\n");
}

void ishmem_tester::print_bw_result(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                    size_t nelems, size_t groups, size_t threads, int pe,
                                    double lat_us, double bw_mb)
{
    assert((pe >= 0) && (pe < n_pes));

    const char *pe_str;
    if (pe == 0) pe_str = "self";
    if (pe == 1) pe_str = "tile";
    if (pe > 1) pe_str = "xe";
    if (csvflag && (my_pe == 0)) {
        printf("csv,%s,%d,%d,%s,%s,%s,%lu,%lu,%lu,%s,%f,%f\n", testname, enable_ipc, n_pes,
               type_to_str(t), op_to_str(op), mode_to_str(mode), groups, threads, nelems, pe_str,
               lat_us, bw_mb);
    } else {
        printf(
            "test %s n_pes %d type %s op %s mode %s groups %lu threads %lu bytes %lu pe %s latency "
            "%f us bw "
            "%f "
            "MB/s\n",
            testname, n_pes, type_to_str(t), op_to_str(op), mode_to_desc(mode), groups, threads,
            nelems, pe_str, lat_us, bw_mb);
    }
}
void ishmem_tester::run_bw_tests(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                 long bandwidth_multiplier, bool collective)
{
    double duration = 0;
    bool tested;
    double thistry;
    size_t iterations;
    size_t max_threads = work_group_size;
    size_t maximum_groups = max_groups;
    if (csvflag) print_bw_header();
    switch (mode) {
        case host_host_host:
        case host_host_device:
        case host_device_host:
        case host_device_device:
        case on_queue:
        case device: {
            max_threads = 1L;
            maximum_groups = 1L;
            break;
        }
        case device_subgroup:
        case device_grp1:
        case device_grp2:
        case device_grp3: {
            maximum_groups = 1L;
            break;
        }
        case device_multi_wg: {
            break;
        }
        default: {
            assert(0);
            break;
        }
    }
    cmd[0].mode = mode;
    cmd[0].op = op;
    cmd[0].type = t;
    if (my_pe == 0) {
        for (size_t nelems = 1; nelems <= max_nelems; nelems <<= 1) {
            cmd[0].nelems = nelems;
            for (size_t threads = 1; (size_t) threads <= max_threads; threads <<= 1) {
                size_t this_max_groups = maximum_groups;
                if ((this_max_groups * threads) > nelems) {
                    this_max_groups = nelems / threads;
                    if (this_max_groups < 1) this_max_groups = 1;
                }
                for (size_t groups = 1; (size_t) groups <= this_max_groups; groups <<= 1) {
                    if (verboseflag) {
                        printf("[%d] Test type %s op %s mode %s threads %ld nelems %ld\n", my_pe,
                               type_to_str(t), op_to_str(op), mode_to_desc(mode), threads, nelems);
                    }
                    cmd[0].groups = groups;
                    cmd[0].threads = threads;
                    iterations = 1;
                    while (iterations <= 16384) {
                        cmd[0].iter = iterations;
                        cmd[0].cmd = (collective) ? cmd_run : cmd_idle;
                        broadcastcmd();
                        duration = do_test_bw(t, op, mode, groups, threads, iterations, nelems);
                        if (duration > 0.002) break;
                        iterations <<= 1;
                    }
                    tested = duration > 0.002;
                    // now iterations is set, get fastest of 10 tries
                    thistry = duration;
                    for (int best = 0; best < 10; best += 1) {
                        cmd[0].cmd = (collective) ? cmd_run : cmd_idle;
                        broadcastcmd();
                        thistry = do_test_bw(t, op, mode, groups, threads, iterations, nelems);
                        if (thistry < duration) duration = thistry;
                    }
                    if (tested) {
                        double lat_us = (duration / (double) iterations) * 1000000.0;
                        double bw_mb = ((double) sizeof(long) * (double) nelems *
                                        (double) iterations * (double) bandwidth_multiplier) /
                                       (duration * 1000000.0);
                        print_bw_result(t, op, mode, nelems, groups, threads, n_pes - 1, lat_us,
                                        bw_mb);
                    }
                }
            }
        }
        cmd[0].cmd = cmd_exit;
        broadcastcmd();
    } else {
        while (1) {
            broadcastcmd();
            if (cmd[1].cmd == cmd_run) {
                duration = do_test_bw(cmd[1].type, cmd[1].op, cmd[1].mode, cmd[1].groups,
                                      cmd[1].threads, cmd[1].iter, cmd[1].nelems);
            } else if (cmd[1].cmd == cmd_print) {
                printf("[%d] threads %lu iterations %lu duration %f usec %f\n", my_pe,
                       cmd[1].threads, cmd[1].iter, duration,
                       1000000 * duration / (double) cmd[1].iter);
            } else if (cmd[1].cmd == cmd_idle) {
                continue;
            } else {
                break;
            }
        }
    }
}

void ishmem_tester::run_bw_tests(long bandwidth_multiplier, bool collective)
{
    for (int mode_index = 0; mode_index < num_test_modes; mode_index += 1) {
        for (int type_index = 0; type_index < num_test_types; type_index += 1) {
            for (int op_index = 0; op_index < num_test_ops; op_index += 1) {
                testmode_t mode = test_modes[mode_index];
                ishmemi_type_t t = test_types[type_index];
                ishmemi_op_t op = test_ops[op_index];
                run_bw_tests(t, op, mode, bandwidth_multiplier, collective);
            }
        }
    }
}

static uint64_t global_errors = 0; /* global so shmem_sum_reduce will work */
static uint64_t my_errors = 0;

int ishmem_tester::finalize_and_report(size_t errors)
{
    /* reduce() errors in order to return from the job */
    my_errors = errors;
    ishmemi_test_runtime->uint64_sum_reduce(&global_errors, &my_errors, 1);
    if (my_pe == 0) printf("[%d] errors %zu, global_errors %lu\n", my_pe, errors, global_errors);
    printf("[%d] %s\n", my_pe, (global_errors) ? "Test FAILED\n" : "Test PASSED\n");
    return (global_errors != 0);
}

const char *ishmem_tester::mode_to_desc(testmode_t mode)
{
    if (mode == testmode_t::host_host_host) return ("host from host to host memory");
    if (mode == testmode_t::host_host_device) return ("host from host to device memory");
    if (mode == testmode_t::host_device_host) return ("host from device to host memory");
    if (mode == testmode_t::host_device_device) return ("host from device to device memory");
    if (mode == testmode_t::on_queue) return ("on queue with device memory");
    if (mode == testmode_t::device) return ("device with device memory");
    if (mode == testmode_t::device_grp1) return ("device group<1> with device memory");
    if (mode == testmode_t::device_grp2) return ("device group<2> with device memory");
    if (mode == testmode_t::device_grp3) return ("device group<3> with device memory");
    if (mode == testmode_t::device_subgroup) return ("device sub_group with device memory");
    if (mode == testmode_t::device_multi_wg) return ("device group<1> multi_wg with device memory");
    return "unknown testmode";
}

const char *ishmem_tester::type_to_str(ishmemi_type_t t)
{
    assert((int) t < (int) ISHMEMI_TYPE_END);
    return (ishmemi_type_str[t]);
}

const char *ishmem_tester::op_to_str(ishmemi_op_t op)
{
    assert((int) op < (int) ISHMEMI_OP_END);
    return (ishmemi_op_str[op]);
}

#define STUB_SINGLE_TESTS                                                                          \
    SYCL_EXTERNAL int do_test_single(ishmemi_type_t t, ishmemi_op_t op, void *dest,                \
                                     const void *src, size_t nelems)                               \
    {                                                                                              \
        return (0);                                                                                \
    }

#define STUB_GROUP_TESTS                                                                           \
    template int do_test_work_group<sycl::group<1>>(ishmemi_type_t t, ishmemi_op_t op, void *dest, \
                                                    const void *src, size_t nelems,                \
                                                    const sycl::group<1> &grp);                    \
    template int do_test_work_group<sycl::group<2>>(ishmemi_type_t t, ishmemi_op_t op, void *dest, \
                                                    const void *src, size_t nelems,                \
                                                    const sycl::group<2> &grp);                    \
    template int do_test_work_group<sycl::group<3>>(ishmemi_type_t t, ishmemi_op_t op, void *dest, \
                                                    const void *src, size_t nelems,                \
                                                    const sycl::group<3> &grp);                    \
    template int do_test_work_group<sycl::sub_group>(ishmemi_type_t t, ishmemi_op_t op,            \
                                                     void *dest, const void *src, size_t nelems,   \
                                                     const sycl::sub_group &grp);                  \
    template <typename Group>                                                                      \
    SYCL_EXTERNAL int do_test_work_group(ishmemi_type_t t, ishmemi_op_t op, void *dest,            \
                                         const void *src, size_t nelems, const Group &grp)         \
    {                                                                                              \
        return (0);                                                                                \
    }

#define STUB_UNIT_TESTS                                                                            \
    STUB_SINGLE_TESTS                                                                              \
    STUB_GROUP_TESTS

#endif  // ifdef TESTMACROS_H
