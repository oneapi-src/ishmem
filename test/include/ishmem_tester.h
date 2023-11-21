/* Copyright (C) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ISHMEM_TESTER_H
#define ISHMEM_TESTER_H

#include <type_traits>  // needed for the enum iterator
#include <ishmem.h>
#include <ishmemx.h>
#include <shmem.h>
#include <getopt.h>
#include "common.h"
#include <unistd.h>

constexpr size_t x_size = 16;
constexpr size_t y_size = 2;
constexpr size_t z_size = 2;

/* Iterator class derived from this question[1] and answer[2] on stackoverflow.com
 * Copyright (C) 2015 Francesco Chemolli[3]
 * Licensed under the CC BY-SA 3.0[4]
 * [1]: https://stackoverflow.com/questions/261963/how-can-i-iterate-over-an-enum
 * [2]: https://stackoverflow.com/a/31836401
 * [3]: https://stackoverflow.com/users/2938538/francesco-chemolli
 * [4]: http://creativecommons.org/licenses/by-sa/3.0/
 */

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
typedef enum MODE {
    host_host_host,
    host_host_device,
    host_device_host,
    host_device_device,
    device,
    device_subgroup,
    device_grp1,
    device_grp2,
    device_grp3,
    mode_count
} testmode_t;

typedef Iterator<testmode_t, testmode_t::host_host_host, testmode_t::device_grp3>
    testmode_t_Iterator;

#define cmd_idle  0
#define cmd_run   1
#define cmd_print 2
#define cmd_exit  3

// PE 0 tells other PEs what to run
struct CMD {
    long cmd;
    size_t iter;
    size_t threads;
    size_t nelems;
    ishmemi_type_t type;
    ishmemi_op_t op;
    testmode_t mode;
};

struct test_data_t {
    bool *test_run;
    long other[4096];
};

sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

class ishmem_tester {
  protected:  // intended for use by subclasses
    double tsc_frequency;
    char *testname;                 /* from argv[0] */
    int enable_ipc;                 /* from ishmemi_params.ENABLE_GPU_IPC */
    long *aligned_source = nullptr; /* source pattern, 64 bit aligned */
    long *host_source = nullptr;    /* operation source, if in host memory */
    long *host_dest = nullptr;      /* operation destination, if in host memory */
    long *host_result = nullptr;    /* host copy of operation actual results */
    long *host_check = nullptr;     /* host copy of expected results */
    void *device_source = nullptr;  /* source buffer, if in device memory */
    void *device_dest = nullptr;    /* destination buffer, if in device memory */
    int *test_return =
        nullptr; /* storage to record return value from test funtion, in host memory */
    test_data_t test_data;
    struct CMD *cmd;
    testmode_t test_modes[(int) mode_count];
    int num_test_modes = 0;

    template <typename T>
    size_t tcheck(T *expected, T *actual, size_t nelems);
    size_t check(ishmemi_type_t t, size_t nelems);
    bool source_is_device(testmode_t mode);
    bool dest_is_device(testmode_t mode);
    void print_bw_header();
    void print_bw_result(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t nelems,
                         size_t threads, int pe, double lat_us, double bw_mb);
    /* return byte size of the source pattern */
    virtual size_t create_source_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                         size_t nelems);
    /* return byte size of the check pattern */
    virtual size_t create_check_pattern(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                        size_t nelems);
    void parse_tester_args(int argc, char *argv[]);  // read command line
    bool parse_test_modes(char *arg);  // build table of modes to test, reports success
  public:                              // intended for use by users
    int my_pe;                         // set by job launch
    int n_pes;                         // set by job launch
    size_t max_nelems = 1L << 16;      // default value, override on command line
    size_t work_group_size = 1024;     // max and default value, override on command line
    int patterndebugflag = 0;          // print patterns, set with command line --patterndebug
    int verboseflag = 0;               // verbose, set with command line --verbose
    int csvflag = 0;                   // csv mode, set with command line --csv

    sycl::queue q;

    ishmem_tester(int argc, char *argv[]) : q(prop_list)
    {
        testname = basename(argv[0]);
        parse_tester_args(argc, argv);
        /* If there was no testmode list on the command line, use a subset */
        if (num_test_modes == 0) {
            test_modes[num_test_modes++] = device;
            test_modes[num_test_modes++] = device_grp1;
        }
        ishmem_init();

        my_pe = ishmem_my_pe();  // global so called functions can use it
        n_pes = ishmem_n_pes();
        enable_ipc = true;
        const char *env_val = getenv("ISHMEM_ENABLE_GPU_IPC");
        if (env_val != nullptr) {
            if (strcasecmp("false", env_val) == 0) enable_ipc = false;
            else if (strcasecmp("0", env_val) == 0) enable_ipc = false;
        }
        tsc_frequency = measure_tsc_frequency();
        cmd = (struct CMD *) shmem_calloc(2, sizeof(struct CMD));
        test_data.test_run = (bool *) sycl::malloc_host(sizeof(bool), q);
        setbuf(stdout, NULL); /* turn off buffering */
        setbuf(stderr, NULL);
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
    double do_test_bw(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t threads,
                      size_t iterations, size_t nelems);
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
    void run_bw_tests(testmode_t mode, long bandwidth_multiplier, bool collective);
    /* this version runs through the mode list set by the -t switch */
    void run_bw_tests(ishmemi_type_t t, ishmemi_op_t op, long bandwidth_multiplier,
                      bool collective);

    /* decode the enum values into strings for printfs */
    const char *modestr(testmode_t mode);
    const char *modename(testmode_t t);
    const char *typestr(ishmemi_type_t t);
    /* returns sizeof() the relevant datatype */
    size_t typesize(ishmemi_type_t t);
};

/* forward declarations for functions to be defined by user.  Typically these are defined by calls
 * on the macros below */

SYCL_EXTERNAL int do_test_single(ishmemi_type_t t, void *dest, const void *src, size_t nelems,
                                 bool *test_run);
template <typename Group>
SYCL_EXTERNAL int do_test_work_group(ishmemi_type_t t, void *dest, const void *src, size_t nelems,
                                     const Group &grp, bool *test_run);

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
 * would work for a function to return the size of each ishmem_type_t
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
            case LONGLONG:                                                                         \
                ISHMEM_TYPE_BRANCH(LONGLONG, longlong, long long)                                  \
            case UCHAR:                                                                            \
                ISHMEM_TYPE_BRANCH(UCHAR, uchar, unsigned char)                                    \
            case USHORT:                                                                           \
                ISHMEM_TYPE_BRANCH(USHORT, ushort, unsigned short)                                 \
            case UINT:                                                                             \
                ISHMEM_TYPE_BRANCH(UINT, uint, unsigned int)                                       \
            case ULONG:                                                                            \
                ISHMEM_TYPE_BRANCH(ULONG, ulong, unsigned long)                                    \
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
            case UINT8:                                                                            \
                ISHMEM_TYPE_BRANCH(UINT8, uint8, uint8_t)                                          \
            case UINT16:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT16, uint16, uint16_t)                                       \
            case UINT32:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT32, uint32, uint32_t)                                       \
            case UINT64:                                                                           \
                ISHMEM_TYPE_BRANCH(UINT64, uint64, uint64_t)                                       \
            case SIZE:                                                                             \
                ISHMEM_TYPE_BRANCH(SIZE, size, size_t)                                             \
            case PTRDIFF:                                                                          \
                ISHMEM_TYPE_BRANCH(PTRDIFF, ptrdiff, ptrdiff_t);                                   \
            case MEM:                                                                              \
                memcase default : return (res);                                                    \
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
            default:                                                                               \
                res = "unknown";                                                                   \
        }                                                                                          \
        return (res);                                                                              \
    }

#define ISHMEM_GEN_TEST_FUNCTION_SINGLE(returnvar, memcase)                                        \
    ISHMEM_GEN_TYPE_FUNCTION(                                                                      \
        int do_test_single(                                                                        \
            ishmemi_type_t t,                                                                      \
            void *dest COMMA const void *src COMMA size_t nelems COMMA bool *test_run),            \
        returnvar, memcase)

#define ISHMEM_GEN_TEST_FUNCTION_WORK_GROUP(returnvar, memcase)                                    \
    ISHMEM_GEN_TYPE_FUNCTION(                                                                      \
        template int do_test_work_group<sycl::group<1>>(                                           \
            ishmemi_type_t t COMMA void *dest COMMA const void *src COMMA size_t nelems            \
                COMMA const sycl::group<1> &grp COMMA bool *test_run);                             \
        template int do_test_work_group<sycl::group<2>>(                                           \
            ishmemi_type_t t COMMA void *dest COMMA const void *src COMMA size_t nelems            \
                COMMA const sycl::group<2> &grp COMMA bool *test_run);                             \
        template int do_test_work_group<sycl::group<3>>(                                           \
            ishmemi_type_t t COMMA void *dest COMMA const void *src COMMA size_t nelems            \
                COMMA const sycl::group<3> &grp COMMA bool *test_run);                             \
        template int do_test_work_group<sycl::sub_group>(                                          \
            ishmemi_type_t t COMMA void *dest COMMA const void *src COMMA size_t nelems            \
                COMMA const sycl::sub_group &grp COMMA bool *test_run);                            \
        template <typename Group> SYCL_EXTERNAL int do_test_work_group(                            \
            ishmemi_type_t t COMMA void *dest COMMA const void *src COMMA size_t nelems            \
                COMMA const Group &grp COMMA bool *test_run),                                      \
        returnvar, memcase)

template <typename Group>
SYCL_EXTERNAL int do_test_work_group(ishmemi_type_t t, void *dest, const void *src, size_t nelems,
                                     const Group &grp, bool *test_run);

/* these are here to keep the compiler happy, if needed */
#ifndef BW_TEST_HEADER
#define BW_TEST_HEADER
#endif
#ifndef BW_TEST_FUNCTION
#define BW_TEST_FUNCTION
#endif
#ifndef BW_TEST_FUNCTION_WORK_GROUP
#define BW_TEST_FUNCTION_WORK_GROUP
#endif

/* this defines a function
const char *ishmem_tester::typestr(ishmemi_type_t t);
which returns the shmem name corresponding to the enum type passed in
*/

#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = #name;                                                                                   \
    break;

ISHMEM_GEN_TYPE_FUNCTION(const char *ishmem_tester::typestr(ishmemi_type_t t),
                         const char *res = nullptr;
                         , res = "mem"; break;)

/* this defines a function
const char *ishmem_tester::modename(testmode_t mode);
which returns the shmem name corresponding to the enum type passed in
*/

#ifdef ISHMEM_MODE_BRANCH
#undef ISHMEM_MODE_BRANCH
#endif

#define ISHMEM_MODE_BRANCH(name)                                                                   \
    res = #name;                                                                                   \
    break;

ISHMEM_GEN_MODE_FUNCTION(const char *ishmem_tester::modename(testmode_t mode))

/* This defines a function
size_t ishmem_tester::typesize(ishmemi_type_t t);
which returns the sizeof(the type corresponding to the enum passed in
*/
#ifdef ISHMEM_TYPE_BRANCH
#undef ISHMEM_TYPE_BRANCH
#endif

#define ISHMEM_TYPE_BRANCH(enumname, name, type)                                                   \
    res = sizeof(type);                                                                            \
    break;

ISHMEM_GEN_TYPE_FUNCTION(size_t ishmem_tester::typesize(ishmemi_type_t t), size_t res = 0;, res = 1;
                         break;)

template <typename T>
size_t ishmem_tester::tcheck(T *expected, T *actual, size_t nelems)
{
    size_t errors = 0;
    for (size_t idx = 0; idx < nelems; idx += 1) {
        if constexpr (std::is_same_v<T, float>) {
            uint32_t got = ((uint32_t *) actual)[idx];
            uint32_t exp = ((uint32_t *) expected)[idx];
            if (got != exp) {
                errors += 1;
                if (errors <= 16) {
                    printf("[%d] err idx 0x%x nelems %ld got %08x expected %08x\n", my_pe,
                           (unsigned int) idx, nelems, (uint32_t) got, (uint32_t) exp);
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            uint64_t got = ((uint64_t *) actual)[idx];
            uint64_t exp = ((uint64_t *) expected)[idx];
            if (got != exp) {
                errors += 1;
                if (errors <= 16) {
                    printf("[%d] err idx 0x%x nelems %ld got %016lx expected %016lx\n", my_pe,
                           (unsigned int) idx, nelems, (uint64_t) got, (uint64_t) exp);
                }
            }
        } else {
            T got = actual[idx];
            T exp = expected[idx];

            if (got != exp) {
                errors += 1;
                if (errors <= 16) {
                    printf("[%d] err idx 0x%x nelems %ld got %016lx expected %016lx\n", my_pe,
                           (unsigned int) idx, nelems, (long) got, (long) exp);
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

ISHMEM_GEN_TYPE_FUNCTION(size_t ishmem_tester::check(ishmemi_type_t t COMMA size_t nelems),
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
                 "  --work_group_size, -w  Set the dimensions of the device kernel's"
                 " entire index space (only used in multithreaded tests)\n"
                 " before measurements begin\n"
                 "  --csv -c          Output in csv format\n"
                 "  --patterndebug -p Print testpatterns\n"
                 "  --test_modes mode[,mode]* | all  Select modes to test\n"
                 "  --verbose -v      Print each test\n"
                 "  --help,  -h       Print usage message\n";
    exit(1);
}

static bool tester_isPowerOfTwo(unsigned long n)
{
    return (__builtin_popcountl(n) == 1);
}

/* input is a string like mode,mode,mode  or "all" */
bool ishmem_tester::parse_test_modes(char *arg)
{
    if (strcmp("all", arg) == 0) {
        num_test_modes = 0;
        for (testmode_t mode : testmode_t_Iterator()) {
            test_modes[num_test_modes++] = mode;
        }
        return true;
    }
    char *saveptr;
    char *m = strtok_r(arg, ",", &saveptr);
    while (m) {
        bool found = false;
        for (testmode_t mode : testmode_t_Iterator()) {
            if (strcmp(modename(mode), m) == 0) {
                if (num_test_modes >= mode_count) {
                    printf("too many test modes\n");
                    return (false);
                }
                test_modes[num_test_modes++] = mode;
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

void ishmem_tester::parse_tester_args(int argc, char *argv[])
{
    static struct option long_opts[] = {{"max_nelems", required_argument, nullptr, 'm'},
                                        {"work_group_size", required_argument, nullptr, 'w'},
                                        {"patterndebug", no_argument, &patterndebugflag, 'p'},
                                        {"verbose", no_argument, &verboseflag, 'v'},
                                        {"csv", no_argument, &csvflag, 'c'},
                                        {"help", required_argument, nullptr, 'h'},
                                        {"test_modes", required_argument, nullptr, 't'},
                                        {0, 0, nullptr, 0}};
    while (true) {
        const auto opt = getopt_long(argc, argv, "m:w:pvcht:", long_opts, nullptr);
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
            case 'h':
            case '?':
                print_tester_usage();
                exit(1);
            default:
                break;
        }
    }
}

void ishmem_tester::alloc_memory(size_t bufsize)
{
    /* allocate host memory */
    aligned_source = (long *) malloc(bufsize); /* data pattern */
    assert(aligned_source != nullptr);

    host_check = (long *) malloc(bufsize); /* expected data */
    assert(host_check != nullptr);
    host_source = (long *) shmem_malloc(bufsize); /* source data for this PE */
    assert(host_source != nullptr);
    host_dest = (long *) shmem_malloc(bufsize); /* source data for this PE */
    assert(host_dest != nullptr);
    host_result = (long *) malloc(bufsize); /* used to read back actual data */
    assert(host_result != nullptr);
    /* allocate GPU memory for source and destination */
    /* the extra 1024 is for the larger offsets, shouldn't be needed */
    device_source = (long *) ishmem_malloc(bufsize); /* gpu source, if used */
    assert(device_source != nullptr);
    device_dest = (long *) ishmem_malloc(bufsize); /* gpu destination, if used */
    assert(device_dest != nullptr);

    test_return = sycl::malloc_host<int>(1, q);
    assert(test_return != nullptr);
}

ishmem_tester::~ishmem_tester()
{
    free(aligned_source);
    free(host_check);
    free(host_result);
    shmem_free(cmd);
    shmem_free(host_source);
    shmem_free(host_dest);
    ishmem_free(device_source);
    ishmem_free(device_dest);
    sycl::free(test_return, q);
    sycl::free(test_data.test_run, q);
    ishmem_finalize();
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

size_t ishmem_tester::do_test(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t nelems,
                              unsigned long source_offset, unsigned long dest_offset)
{
    void *test_source = (source_is_device(mode)) ? device_source : host_source;
    void *test_dest = (dest_is_device(mode)) ? device_dest : host_dest;
    if (t == LONGDOUBLE) return (0); /* do not report errors here */

    size_t source_size = create_source_pattern(t, op, mode, nelems);
    size_t check_size = create_check_pattern(t, op, mode, nelems);

    bool *test_run = test_data.test_run;
    *test_run = true;

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
        q.memset(test_dest, 128 + my_pe, check_size).wait_and_throw(); /* prefill destination */
    } else {
        memset(test_dest, 128 + my_pe, check_size); /* prefill destination */
    }
    switch (mode) {
        case host_host_host:
        case host_host_device:
        case host_device_host:
        case host_device_device: {
            ishmem_sync_all();
            *local_test_return = do_test_single(t, test_dest, test_source, nelems, test_run);
            ishmem_sync_all();
            break;
        }
        case device: {
            q.single_task([=]() {
                 ishmem_sync_all();
                 *local_test_return = do_test_single(t, test_dest, test_source, nelems, test_run);
                 ishmem_sync_all();
             }).wait_and_throw();
            break;
        }
        case device_subgroup: {
            q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),
                           [=](sycl::nd_item<1> it) {
                               auto grp = it.get_sub_group();
                               ishmemx_sync_all_work_group(grp);
                               *local_test_return = do_test_work_group(t, test_dest, test_source,
                                                                       nelems, grp, test_run);
                               ishmemx_sync_all_work_group(grp);
                           })
                .wait_and_throw();
            break;
        }
        case device_grp1: {
            q.parallel_for(sycl::nd_range<1>(sycl::range<1>(x_size), sycl::range<1>(x_size)),
                           [=](sycl::nd_item<1> it) {
                               auto grp = it.get_group();
                               ishmemx_sync_all_work_group(grp);
                               *local_test_return = do_test_work_group(t, test_dest, test_source,
                                                                       nelems, grp, test_run);
                               ishmemx_sync_all_work_group(grp);
                           })
                .wait_and_throw();
            break;
        }
        case device_grp2: {
            q.parallel_for(
                 sycl::nd_range<2>(sycl::range<2>(x_size, y_size), sycl::range<2>(x_size, y_size)),
                 [=](sycl::nd_item<2> it) {
                     auto grp = it.get_group();
                     ishmemx_sync_all_work_group(grp);
                     *local_test_return =
                         do_test_work_group(t, test_dest, test_source, nelems, grp, test_run);
                     ishmemx_sync_all_work_group(grp);
                 })
                .wait_and_throw();
            break;
        }
        case device_grp3: {
            q.parallel_for(sycl::nd_range<3>(sycl::range<3>(x_size, y_size, z_size),
                                             sycl::range<3>(x_size, y_size, z_size)),
                           [=](sycl::nd_item<3> it) {
                               auto grp = it.get_group();
                               ishmemx_sync_all_work_group(grp);
                               *local_test_return = do_test_work_group(t, test_dest, test_source,
                                                                       nelems, grp, test_run);
                               ishmemx_sync_all_work_group(grp);
                           })
                .wait_and_throw();
            break;
        }
        default:
            assert(0);
            break;
    }
    if (*local_test_return != 0) {
        printf("[%d] Test %s datatype %s op %s nelems %ld FAIL return value %d\n", my_pe,
               modestr(mode), typestr(t), ishmemi_op_str[op], nelems, *local_test_return);
    }
    size_t errors = 0;
    if (dest_is_device(mode)) q.memcpy(host_result, test_dest, check_size).wait_and_throw();
    else memcpy(host_result, test_dest, check_size);
    /* check routine compares host_result with host_check */
    if (*test_run) errors = check(t, check_size / typesize(t));
    if (errors > 0) {
        printf("[%d] Test %s datatype %s op %s nelems %ld errors %ld\n", my_pe, modestr(mode),
               typestr(t), ishmemi_op_str[op], nelems, errors);
    }

    return (errors);
}

double ishmem_tester::do_test_bw(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode, size_t threads,
                                 size_t iterations, size_t nelems)
{
    void *src __attribute__((unused)) = nullptr;
    void *dest __attribute__((unused)) = nullptr;

    src = (source_is_device(mode)) ? device_source : host_source;
    dest = (dest_is_device(mode)) ? device_dest : host_dest;
    double duration = 0;
    BW_TEST_HEADER;  // define any needed variables
    switch (mode) {
        case host_host_host:
        case host_host_device:
        case host_device_host:
        case host_device_device: {
            unsigned long start = rdtsc();
            BW_TEST_FUNCTION;
            unsigned long stop = rdtsc();
            duration = ((double) (stop - start)) / tsc_frequency;
            break;
        }
        case device: {
            auto e = q.single_task([=]() { BW_TEST_FUNCTION; });
            e.wait_and_throw();
            duration = getduration(e);
            break;
        }
        case device_subgroup:
        case device_grp3:
        case device_grp2:
        case device_grp1: {
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
        default:
            assert(0);
            break;
    }
    return (duration);
}

double ishmem_tester::do_test_bw(testmode_t mode, size_t threads, size_t iterations, size_t nelems)
{
    return (do_test_bw(LONG, NOP, mode, threads, iterations, nelems));
}

typedef Iterator<ishmemi_type_t, ishmemi_type_t::FLOAT, ishmemi_type_t::MEM>
    ishmemi_type_t_Iterator;

size_t ishmem_tester::run_aligned_tests(ishmemi_op_t op)
{
    size_t errors = 0;
    printf("[%d] Run Aligned Tests op %s\n", my_pe, ishmemi_op_str[op]);
    for (int i = 0; i < num_test_modes; i += 1) {
        testmode_t mode = test_modes[i];
        printf("[%d] Testing %s\n", my_pe, modestr(mode));
        fflush(stdout);
        /* test all datatypes */
        for (ishmemi_type_t t : ishmemi_type_t_Iterator()) {
            /* test power of two sizes */
            for (size_t nelems = 1; nelems <= max_nelems; nelems <<= 1) {
                if (verboseflag && (my_pe == 0)) {
                    printf("[%d] Test %s %s %s nelems %ld os %ld od %ld\n", my_pe, modestr(mode),
                           typestr(t), ishmemi_op_str[op], nelems, 0L, 0L);
                    fflush(stdout);
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
    return (run_aligned_tests(NOP));
}

size_t ishmem_tester::run_offset_tests(ishmemi_op_t op)
{
    size_t errors = 0;
    /* quick tests of different source and destination offsets and small lengths */
    /* could be sped up by making the numbers of cases datatype dependent */
    printf("[%d] Run Offset Tests op %s\n", my_pe, ishmemi_op_str[op]);
    for (int i = 0; i < num_test_modes; i += 1) {
        testmode_t mode = test_modes[i];
        printf("[%d] Testing %s\n", my_pe, modestr(mode));
        fflush(stdout);
        for (ishmemi_type_t t : ishmemi_type_t_Iterator()) {
            for (size_t nelems = 1; nelems <= 16; nelems += 1) {
                /* offsets run from 0 to 15 in units of the datatype size */
                for (unsigned long source_offset = 0; source_offset < 15;
                     source_offset += typesize(t)) {
                    for (unsigned long dest_offset = 0; dest_offset < 15;
                         dest_offset += typesize(t)) {
                        if (verboseflag && (my_pe == 0)) {
                            printf("[%d] Test %s %s nelems %ld os %ld od %ld\n", my_pe,
                                   modestr(mode), typestr(t), nelems, source_offset, dest_offset);
                            fflush(stdout);
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
    return (run_offset_tests(NOP));
}

void ishmem_tester::print_bw_header()
{
    if (csvflag && (my_pe == 0))
        printf("csv,testname,ipc,npes,type,op,mode,threads,nelems,pe,latency_us,bw_mb\n");
}

void ishmem_tester::print_bw_result(ishmemi_type_t t, ishmemi_op_t op, testmode_t mode,
                                    size_t nelems, size_t threads, int pe, double lat_us,
                                    double bw_mb)
{
    const char *pe_str;
    if (pe == 0) pe_str = "self";  // this is really hacky and job dependent
    if (pe == 1) pe_str = "tile";
    if (pe > 1) pe_str = "xe";
    if (csvflag && (my_pe == 0)) {
        printf("csv,%s,%d,%d,%s,%s,%s,%lu,%lu,%s,%f,%f\n", testname, enable_ipc, n_pes, typestr(t),
               ishmemi_op_str[op], modename(mode), threads, nelems, pe_str, lat_us, bw_mb);
    } else {
        printf(
            "test %s n_pes %d type %s op %s mode %s threads %lu nelems %lu pe %s latency %f us bw "
            "%f "
            "MB/s\n",
            testname, n_pes, typestr(t), ishmemi_op_str[op], modestr(mode), threads, nelems, pe_str,
            lat_us, bw_mb);
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
    if (csvflag) print_bw_header();
    if (mode == host_host_host) max_threads = 1L;
    if (mode == host_host_device) max_threads = 1L;
    if (mode == host_device_host) max_threads = 1L;
    if (mode == host_device_device) max_threads = 1L;
    if (mode == device) max_threads = 1L;
    cmd[0].mode = mode;
    cmd[0].op = op;
    cmd[0].type = t;
    if (my_pe == 0) {
        for (size_t nelems = 1; nelems <= max_nelems; nelems <<= 1) {
            cmd[0].nelems = nelems;
            for (size_t threads = 1; (size_t) threads <= max_threads; threads <<= 1) {
                if (verboseflag) {
                    printf("[%d] Test type %s op %s mode %s threads %ld nelems %ld\n", my_pe,
                           typestr(t), ishmemi_op_str[op], modestr(mode), threads, nelems);
                }
                cmd[0].threads = threads;
                iterations = 1;
                while (iterations <= 16384) {
                    cmd[0].iter = iterations;
                    cmd[0].cmd = (collective) ? cmd_run : cmd_idle;
                    shmem_broadcastmem(SHMEM_TEAM_WORLD, &cmd[1], &cmd[0], sizeof(struct CMD), 0);
                    duration = do_test_bw(t, op, mode, threads, iterations, nelems);
                    if (duration > 0.002) break;
                    iterations <<= 1;
                }
                tested = duration > 0.002;
                // now iterations is set, get fastest of 10 tries
                thistry = duration;
                for (int best = 0; best < 10; best += 1) {
                    cmd[0].cmd = (collective) ? cmd_run : cmd_idle;
                    shmem_broadcastmem(SHMEM_TEAM_WORLD, &cmd[1], &cmd[0], sizeof(struct CMD), 0);
                    thistry = do_test_bw(t, op, mode, threads, iterations, nelems);
                    if (thistry < duration) duration = thistry;
                }
                if (tested) {
                    double lat_us = (duration / (double) iterations) * 1000000.0;
                    double bw_mb = ((double) sizeof(long) * (double) nelems * (double) iterations *
                                    (double) bandwidth_multiplier) /
                                   (duration * 1000000.0);
                    print_bw_result(t, op, mode, nelems, threads, n_pes - 1, lat_us, bw_mb);
                }
            }
        }
        cmd[0].cmd = cmd_exit;
        shmem_broadcastmem(SHMEM_TEAM_WORLD, &cmd[1], &cmd[0], sizeof(struct CMD), 0);
    } else {
        while (1) {
            shmem_broadcastmem(SHMEM_TEAM_WORLD, &cmd[1], &cmd[0], sizeof(struct CMD), 0);
            if (cmd[1].cmd == cmd_run) {
                duration = do_test_bw(cmd[1].type, cmd[1].op, cmd[1].mode, cmd[1].threads,
                                      cmd[1].iter, cmd[1].nelems);
            } else if (cmd[1].cmd == cmd_print) {
                printf("[%d] threads %lu iterations %lu duration %f usec %f\n", my_pe,
                       cmd[1].threads, cmd[1].iter, duration,
                       1000000 * duration / (double) cmd[1].iter);
                fflush(stdout);
            } else if (cmd[1].cmd == cmd_idle) {
                continue;
            } else {
                break;
            }
        }
    }
}

void ishmem_tester::run_bw_tests(ishmemi_type_t t, ishmemi_op_t op, long bandwidth_multiplier,
                                 bool collective)
{
    for (int i = 0; i < num_test_modes; i += 1) {
        testmode_t mode = test_modes[i];
        run_bw_tests(t, op, mode, bandwidth_multiplier, collective);
    }
}

void ishmem_tester::run_bw_tests(testmode_t mode, long bandwidth_multiplier, bool collective)
{
    run_bw_tests(LONG, NOP, mode, bandwidth_multiplier, collective);
}

static size_t global_errors = 0; /* global so shmem_sum_reduce will work */
static size_t my_errors = 0;

int ishmem_tester::finalize_and_report(size_t errors)
{
    /* reduce() errors in order to return from the job */
    my_errors = errors;
    shmem_sum_reduce(SHMEM_TEAM_WORLD, &global_errors, &my_errors, 1);
    if (my_pe == 0) printf("[%d] errors %ld, global_errors %ld\n", my_pe, errors, global_errors);
    printf("[%d] %s\n", my_pe, (global_errors) ? "Test FAILED\n" : "Test PASSED\n");
    return (global_errors != 0);
}

const char *ishmem_tester::modestr(testmode_t mode)
{
    if (mode == testmode_t::host_host_host) return ("host from host to host memory");
    if (mode == testmode_t::host_host_device) return ("host from host to device memory");
    if (mode == testmode_t::host_device_host) return ("host from device to host memory");
    if (mode == testmode_t::host_device_device) return ("host from device to device memory");
    if (mode == testmode_t::device) return ("device with device memory");
    if (mode == testmode_t::device_grp1) return ("device group<1> with device memory");
    if (mode == testmode_t::device_grp2) return ("device group<2> with device memory");
    if (mode == testmode_t::device_grp3) return ("device group<3> with device memory");
    if (mode == testmode_t::device_subgroup) return ("device sub_group with device memory");
    return "unknown testmode";
}

#define STUB_UNIT_TESTS                                                                            \
    SYCL_EXTERNAL int do_test_single(ishmemi_type_t t, void *dest, const void *src, size_t nelems, \
                                     bool *test_run)                                               \
    {                                                                                              \
        return (0);                                                                                \
    }                                                                                              \
                                                                                                   \
    template int do_test_work_group<sycl::group<1>>(ishmemi_type_t t, void *dest, const void *src, \
                                                    size_t nelems, const sycl::group<1> &grp,      \
                                                    bool *test_run);                               \
    template int do_test_work_group<sycl::group<2>>(ishmemi_type_t t, void *dest, const void *src, \
                                                    size_t nelems, const sycl::group<2> &grp,      \
                                                    bool *test_run);                               \
    template int do_test_work_group<sycl::group<3>>(ishmemi_type_t t, void *dest, const void *src, \
                                                    size_t nelems, const sycl::group<3> &grp,      \
                                                    bool *test_run);                               \
    template int do_test_work_group<sycl::sub_group>(ishmemi_type_t t, void *dest,                 \
                                                     const void *src, size_t nelems,               \
                                                     const sycl::sub_group &grp, bool *test_run);  \
    template <typename Group>                                                                      \
    SYCL_EXTERNAL int do_test_work_group(ishmemi_type_t t, void *dest, const void *src,            \
                                         size_t nelems, const Group &grp, bool *test_run)          \
    {                                                                                              \
        return (0);                                                                                \
    }

#endif  // ifdef TESTMACROS_H
