import { Router } from "express";
import { v4 as uuid } from "uuid";

import { createJob, getJob, getAllJobs } from "./jobManager.js";
import { launchWorker } from "./workerLauncher.js";

const router = Router();

router.post("/train", async (req, res) => {
  const workers = req.body.workers ?? 3;

  const jobId = uuid();

  const partitions = [];

  const workerList = [];

  for (let i = 0; i < workers; i++) {
    partitions.push(i);
  }

  const job = {
    id: jobId,
    status: "LAUNCHING_WORKERS",
    workers: workerList,
    partitions,
    createdAt: Date.now(),
    results: [],
  };

  createJob(job);

  for (const partition of partitions) {
    const podName = await launchWorker(jobId, partition);

    workerList.push({
      id: partition.toString(),
      partition,
      podName,
      status: "PENDING",
    });
  }

  job.status = "TRAINING";

  res.json({
    message: "Training Started",
    jobId,
  });
});

router.get("/jobs", (req, res) => {
  res.json(getAllJobs());
});

router.get("/jobs/:id", (req, res) => {
  const job = getJob(req.params.id);

  if (!job) {
    return res.status(404).json({
      error: "Job not found",
    });
  }

  res.json(job);
});

export default router;
