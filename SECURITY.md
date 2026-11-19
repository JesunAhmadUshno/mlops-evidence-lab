# Security policy

Report a vulnerability privately through GitHub's "Report a
vulnerability" button on this repository's Security tab. Please do not
open a public issue for it. You will get a reply within 7 days.

Scope: the code in this repository. MLflow runs on a local sqlite
store with no server. The Gemini judge credential is attached at call
time by a connector helper outside this repository; no credential,
key or token is stored here, and the call log records request status
only.
