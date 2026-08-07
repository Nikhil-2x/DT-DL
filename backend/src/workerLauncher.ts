import { coreApi } from "./kubernetes.js";

export async function launchWorker(jobId: string, partition: number) {
  const podName = `job-${jobId.slice(0, 8)}-worker-${partition}`;

  const pod = {
    apiVersion: "v1",

    kind: "Pod",

    metadata: {
      name: podName,

      labels: {
        jobId,

        worker: partition.toString(),
      },
    },

    spec: {
      restartPolicy: "Never",
      containers: [
        {
          name: "trainer",
          image: "localhostnick/trainer",
          imagePullPolicy: "Always",
          env: [
            {
              name: "JOB_ID",
              value: jobId,
            },
            {
              name: "PARTITION",
              value: partition.toString(),
            },
          ],
        },
      ],
    },
  };

  await coreApi.createNamespacedPod({
    namespace: "default",
    body: pod,
  });

  return podName;
}
