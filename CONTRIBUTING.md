# Contributions are welcome!

We're welcoming all contributions from new samples to simple feedback.

## General guideline

* Please keep all samples simple. This project is strongly focusing on easy-to-use and easy-to-understand examples.
* Please try to run each code that you modify before sending a pull request because now we don't setup any CI/CD environment. It is important to check if the codes are running well by yourself.
* It's highly recommended to use NGC container image to test all codes.

## Commits and PRs

Send your PRs to the `main` branch from your forkced branch.

1. Make sure your PR does one thing. Have a clear answer to "What does this PR include?".
2. Read and follow General guideline above.
3. Make sure you sign your commits (e.g., use `git commit -s`) when before your commit.
    - Signing process details are described below.
4. Make sure if all your modifications can work well.
5. Send your PR and request a review.

### Sign Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.
    - Any contribution which contains commits that are not Signed-Off will not be accepted.
* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```
* Full text of the DCO:
  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

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
