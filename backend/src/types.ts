export type WorkerStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED";

export interface Worker {
  id: string;
  partition: number;
  podName: string;
  status: WorkerStatus;
}

export interface WorkerResult {
  worker: string;
  logs: string;
}

export interface Job {
  id: string;
  status:
    | "CREATED"
    | "SPLITTING_DATASET"
    | "LAUNCHING_WORKERS"
    | "TRAINING"
    | "COLLECTING_RESULTS"
    | "COMPLETED"
    | "FAILED";
  workers: Worker[];
  partitions: number[];
  createdAt: number;
  results: WorkerResult[];
}
