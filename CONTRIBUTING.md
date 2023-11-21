# Contributing guidelines

We welcome community contributions to Intel® SHMEM. You can:

- Submit your changes directly with a [pull request](pulls).
- Log a bug or feedback with an [issue](issues).

Refer to our guidelines on [pull requests](#pull-requests) and [issues](#issues) before proceeding.

## Issues

Use [Github issues](issues) to:
- report an issue
- provide feedback
- make a feature request

**Note**: To report a vulnerability, please refer to the [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull requests

Before you submit a pull request, please ensure the following:
- Follow our [code contribution guidelines](#code-contribution-guidelines) and [coding style guidelines](#coding-style-guidelines)
- Extend the existing [unit tests](#unit-tests) when fixing an issue.
- [Signed-off](#sign-your-work) your work.

**Note:** This project follows the [Github flow](https://docs.github.com/en/get-started/quickstart/github-flow) process. To get started with a pull request, see [Github how-to](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

### Code contribution guidelines

Contributed code must be:
- *Tested*
- *Documented*
- *Portable*

### Coding style

The code style and consistency is maintained using `clang-format`. When submitting a contribution, please make sure that it adheres to the existing coding style by using the following command:

```
clang-format -style=file -i <FILE>
```

This will format the code using the `.clang-format` file found in the top-level directory of this repository.

### Unit tests

Be sure to extend the existing tests when fixing an issue.

### Sign your work

Use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. If you can certify the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then add a line to every git commit message:

    Signed-off-by: Kris Smith <kris.smith@email.com>

**Note**: Use your real name.

If you set your `user.name` and `user.email` git configurations, your commit can be signed automatically with `git commit -s` or `git commit -S`.
