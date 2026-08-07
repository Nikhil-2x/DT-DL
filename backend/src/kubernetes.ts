import * as k8s from "@kubernetes/client-node";

const kc = new k8s.KubeConfig();

// Uses ~/.kube/config
kc.loadFromDefault();

export const coreApi = kc.makeApiClient(k8s.CoreV1Api);

export const logClient = new k8s.Log(kc);
