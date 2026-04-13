# Postgres Queue

Review of the blog: https://planetscale.com/blog/keeping-a-postgres-queue-healthy

## Key problem
- while postgres can be used as a queue, there are scenarios where this fails.
- In a transaction, when a row is marked for deletion, it is not deleted immediately. instead, it is marked as a _dead tuple_, and is later cleaned by the vaccum process.
- HOWEVER, SELECT statements still spent time processing all records, and then decide to skip records that are marked as _dead_. This costs adds linearly if the vaccum process cannot keep up with the rate of production of jobs.
- MVCC Horizon -> determines when the vaccum will be run. A long-running transction can hold the horizon longer, which can cause delay on autovaccum process, causing dead tuples to collect, and leading to eventual failure.
- Reproduction of the [postgres job queue and failure by mvcc](https://brandur.org/postgres-queues) shows that while the problem has been mitigated by newer postgres version, it can still be repoduced when rate of jobs(500jobs/sec) is reached.
- Consumers use `FOR UPDATE SKIP LOCKED` to obtain rows that are not being used/seen by other transactions, thereby helping the consumer only process relevant jobs and prevent duplicate processing.


## Solution

- **Resource budgets via traffic control postgres extension** to control how many resources a targetted query has access to
  -  Server share and burst limit: a percentage of server resources and how quickly they can be consumed.
  -  per-query limit: the time a query can run, measured in seocnds of full server usage.
  -  maximum concurrent workers: a percentage of available worker processes.
- Once these budgets are in place, some queries will be cancelled/failed if they exceed the budget. ** It is mandatory for our application to hence support retries, and for it to have a way to _retry_ at a _better_ time i.e a time when the chances of the query failing again are ideally lower).
- Targetted queries include a SQLCommentor tag(eg: action=analytics)
- `idle_in_transaction_session_timeout` can catch and kill "long-runner" idle transaction.
- With the **traffic control**, the issue is fixed even when the number of jobs is moved to 800 jobs/sec, overlapping analytics queries, and using a single worker i.e only a single query can run at any given time.
