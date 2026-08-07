import { coreApi, logClient } from "./kubernetes.js";
import { jobs } from "./jobManager.js";

import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

async function readLogs(podName: string): Promise<string> {
  const { stdout } = await execAsync(`kubectl logs ${podName}`);

  return stdout;
}

// async function readLogs(podName: string): Promise<string> {
//   let logs = "";
//   const stream = new (await import("stream")).PassThrough();
//
//   stream.on("data", (chunk) => {
//     logs += chunk.toString();
//   });
//
//   await logClient.log("default", podName, "trainer", stream);
//
//   return logs;
// }

export function startPodMonitor() {
  setInterval(async () => {
    for (const job of jobs.values()) {
      let completedWorkers = 0;

      for (const worker of job.workers) {
        try {
          const pod = await coreApi.readNamespacedPod({
            name: worker.podName,
            namespace: "default",
          });

          const phase = pod.status?.phase;

          if (phase === "Pending") {
            worker.status = "PENDING";
          } else if (phase === "Running") {
            worker.status = "RUNNING";

            job.status = "TRAINING";
          } else if (phase === "Succeeded") {
            worker.status = "SUCCEEDED";

            completedWorkers++;

            const alreadyCollected = job.results.find(
              (r) => r.worker === worker.id,
            );

            if (!alreadyCollected) {
              const logs = await readLogs(worker.podName);

              job.results.push({
                worker: worker.id,

                logs,
              });
            }
          } else if (phase === "Failed") {
            worker.status = "FAILED";

            job.status = "FAILED";
          }
        } catch (err) {
          console.log(err);
        }
      }

      if (completedWorkers === job.workers.length && job.workers.length > 0) {
        job.status = "COMPLETED";
      }
    }
  }, 3000);
}
