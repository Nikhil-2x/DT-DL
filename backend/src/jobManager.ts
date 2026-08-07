import { Job } from "./types.js";

export const jobs = new Map<string, Job>();

export function createJob(job: Job) {
  jobs.set(job.id, job);
}

export function getJob(id: string) {
  return jobs.get(id);
}

export function getAllJobs() {
  return [...jobs.values()];
}

export function updateJob(id: string, job: Job) {
  jobs.set(id, job);
}
