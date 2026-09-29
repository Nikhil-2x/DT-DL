package handlers_test

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/http/handlers"
	"dtdl/backend/internal/job"
	"dtdl/backend/internal/repository"
	"dtdl/backend/internal/runner"
	"dtdl/backend/internal/storage"
)

type env struct {
	srv   *httptest.Server
	jobs  *job.Service
	store *storage.Memory
	mock  *runner.Mock
	logs  *bytes.Buffer
}

func newEnv(t *testing.T, maxUpload int64) *env {
	t.Helper()
	db, err := repository.Open("sqlite::memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	logs := &bytes.Buffer{}
	log := slog.New(slog.NewTextHandler(logs, nil))
	store := storage.NewMemory()
	mock := runner.NewMock(time.Hour) // never finishes on its own
	ds := dataset.NewService(repository.NewDatasetRepo(db), store, "datasets", maxUpload, log)
	js := job.NewService(repository.NewJobRepo(db), ds, mock, job.Config{
		Image: "ml-runner:dev", Entrypoint: "train.py", CheckpointBucket: "checkpoints", MaxWorkers: 8,
	}, log)
	ds.SetUsageChecker(js)

	srv := httptest.NewServer(handlers.NewRouter(handlers.Deps{Datasets: ds, Jobs: js, MaxUploadBytes: maxUpload, Log: log}))
	t.Cleanup(srv.Close)
	return &env{srv, js, store, mock, logs}
}

func (e *env) do(t *testing.T, method, path, contentType string, body io.Reader) (int, http.Header, []byte) {
	t.Helper()
	req, _ := http.NewRequest(method, e.srv.URL+path, body)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, resp.Header, b
}

func (e *env) json(t *testing.T, method, path, body string) (int, map[string]any) {
	t.Helper()
	var r io.Reader
	if body != "" {
		r = strings.NewReader(body)
	}
	code, _, b := e.do(t, method, path, "application/json", r)
	var out map[string]any
	_ = json.Unmarshal(b, &out)
	return code, out
}

func multipartBody(t *testing.T, fields map[string]string, filename, content string) (string, io.Reader) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for k, v := range fields {
		w.WriteField(k, v)
	}
	if filename != "" {
		fw, _ := w.CreateFormFile("file", filename)
		io.WriteString(fw, content)
	}
	w.Close()
	return w.FormDataContentType(), &buf
}

func (e *env) upload(t *testing.T, name, filename, content string) (int, map[string]any) {
	t.Helper()
	fields := map[string]string{}
	if name != "" {
		fields["name"] = name
	}
	ct, body := multipartBody(t, fields, filename, content)
	code, _, b := e.do(t, "POST", "/api/v1/datasets", ct, body)
	var out map[string]any
	_ = json.Unmarshal(b, &out)
	return code, out
}

func errCode(m map[string]any) string {
	e, _ := m["error"].(map[string]any)
	c, _ := e["code"].(string)
	return c
}

func TestHealth(t *testing.T) {
	e := newEnv(t, 1<<20)
	code, out := e.json(t, "GET", "/api/v1/health", "")
	if code != 200 || out["status"] != "ok" {
		t.Fatalf("%d %v", code, out)
	}
}

func TestRequestIDAndAccessLog(t *testing.T) {
	e := newEnv(t, 1<<20)
	_, hdr, _ := e.do(t, "GET", "/api/v1/health", "", nil)
	id := hdr.Get("X-Request-Id")
	if id == "" {
		t.Fatal("missing X-Request-Id")
	}
	for _, want := range []string{"request_id=" + id, "method=GET", "path=/api/v1/health", "status=200", "duration_ms="} {
		if !strings.Contains(e.logs.String(), want) {
			t.Errorf("access log missing %q:\n%s", want, e.logs.String())
		}
	}
}

func TestUnknownRouteReturnsJSONError(t *testing.T) {
	e := newEnv(t, 1<<20)
	code, out := e.json(t, "GET", "/api/v1/nope", "")
	if code != 404 || errCode(out) != "ROUTE_NOT_FOUND" {
		t.Fatalf("%d %v", code, out)
	}
}

func TestDatasetLifecycle(t *testing.T) {
	e := newEnv(t, 1<<20)

	code, ds := e.upload(t, "iris", "iris.csv", "a,b\n1,2\n")
	if code != 201 || ds["status"] != "READY" || ds["name"] != "iris" || ds["size_bytes"].(float64) != 8 {
		t.Fatalf("%d %v", code, ds)
	}
	id := ds["id"].(string)
	if _, leaked := ds["access_key"]; leaked {
		t.Fatal("credentials must never appear in responses")
	}

	code, list := e.json(t, "GET", "/api/v1/datasets", "")
	if code != 200 || len(list["datasets"].([]any)) != 1 {
		t.Fatalf("%d %v", code, list)
	}
	code, got := e.json(t, "GET", "/api/v1/datasets/"+id, "")
	if code != 200 || got["id"] != id {
		t.Fatalf("%d %v", code, got)
	}

	code, _, _ = e.do(t, "DELETE", "/api/v1/datasets/"+id, "", nil)
	if code != 204 {
		t.Fatalf("delete: %d", code)
	}
	code, got = e.json(t, "GET", "/api/v1/datasets/"+id, "")
	if code != 404 || errCode(got) != "DATASET_NOT_FOUND" {
		t.Fatalf("%d %v", code, got)
	}
	if objs, _ := e.store.List(t.Context(), "datasets", id+"/"); len(objs) != 0 {
		t.Fatal("object should be deleted with the dataset")
	}
}

func TestUploadErrors(t *testing.T) {
	e := newEnv(t, 100)

	code, out := e.json(t, "POST", "/api/v1/datasets", `{"not":"multipart"}`)
	if code != 400 || errCode(out) != "INVALID_MULTIPART" {
		t.Fatalf("%d %v", code, out)
	}
	ct, body := multipartBody(t, map[string]string{"name": "x"}, "", "")
	c, _, b := e.do(t, "POST", "/api/v1/datasets", ct, body)
	if c != 400 || !strings.Contains(string(b), "MISSING_FILE") {
		t.Fatalf("%d %s", c, b)
	}
	if code, out := e.upload(t, "", "big.bin", strings.Repeat("x", 500)); code != 413 || errCode(out) != "UPLOAD_TOO_LARGE" {
		t.Fatalf("%d %v", code, out)
	}
	if code, out := e.upload(t, "", "empty.bin", ""); code != 400 || errCode(out) != "EMPTY_FILE" {
		t.Fatalf("%d %v", code, out)
	}
	// A traversal filename must be neutralised, not stored as given.
	code, ds := e.upload(t, "", "../../etc/passwd", "root")
	if code != 201 || !strings.HasSuffix(ds["key"].(string), "/passwd") || strings.Contains(ds["key"].(string), "..") {
		t.Fatalf("%d %v", code, ds)
	}
	if code, out := e.json(t, "GET", "/api/v1/datasets/nope", ""); code != 404 || errCode(out) != "DATASET_NOT_FOUND" {
		t.Fatalf("%d %v", code, out)
	}
}

func TestJobLifecycle(t *testing.T) {
	e := newEnv(t, 1<<20)

	code, j := e.json(t, "POST", "/api/v1/jobs", `{"name":"cifar-demo","epochs":3,"workers":4}`)
	if code != 201 || j["status"] != "PENDING" || j["workers"].(float64) != 4 {
		t.Fatalf("%d %v", code, j)
	}
	id := j["id"].(string)

	if code, out := e.json(t, "GET", "/api/v1/jobs", ""); code != 200 || len(out["jobs"].([]any)) != 1 {
		t.Fatalf("%d %v", code, out)
	}

	code, j = e.json(t, "POST", "/api/v1/jobs/"+id+"/start", "")
	if code != 200 || j["status"] != "RUNNING" || j["started_at"] == nil {
		t.Fatalf("%d %v", code, j)
	}
	if code, out := e.json(t, "POST", "/api/v1/jobs/"+id+"/start", ""); code != 409 || errCode(out) != "INVALID_JOB_STATE" {
		t.Fatalf("restart: %d %v", code, out)
	}

	code, _, logs := e.do(t, "GET", "/api/v1/jobs/"+id+"/logs?tail=10", "", nil)
	if code != 200 || !strings.Contains(string(logs), "[mock] starting 4 worker(s)") {
		t.Fatalf("%d %s", code, logs)
	}
	if code, out := e.json(t, "GET", "/api/v1/jobs/"+id+"/logs?tail=abc", ""); code != 400 || errCode(out) != "INVALID_TAIL" {
		t.Fatalf("%d %v", code, out)
	}

	code, j = e.json(t, "POST", "/api/v1/jobs/"+id+"/stop", "")
	if code != 202 || j["status"] != "STOPPING" {
		t.Fatalf("%d %v", code, j)
	}
	if err := e.jobs.Reconcile(t.Context()); err != nil {
		t.Fatal(err)
	}
	code, j = e.json(t, "GET", "/api/v1/jobs/"+id, "")
	if code != 200 || j["status"] != "STOPPED" || j["finished_at"] == nil {
		t.Fatalf("%d %v", code, j)
	}
	if !strings.Contains(e.logs.String(), "job_id="+id) || !strings.Contains(e.logs.String(), "event=job_stopped") {
		t.Fatalf("job events missing from logs:\n%s", e.logs.String())
	}
}

func TestJobWithDataset(t *testing.T) {
	e := newEnv(t, 1<<20)
	_, ds := e.upload(t, "d", "d.csv", "x")
	body := `{"name":"with-data","epochs":1,"dataset_id":"` + ds["id"].(string) + `"}`
	code, j := e.json(t, "POST", "/api/v1/jobs", body)
	if code != 201 || j["data_mode"] != "sharded" || j["dataset_id"] != ds["id"] {
		t.Fatalf("%d %v", code, j)
	}
	if code, out := e.json(t, "DELETE", "/api/v1/datasets/"+ds["id"].(string), ""); code != 409 || errCode(out) != "DATASET_IN_USE" {
		t.Fatalf("%d %v", code, out)
	}
}

func TestInvalidJobRequests(t *testing.T) {
	e := newEnv(t, 1<<20)
	cases := []struct {
		name, body, want string
		status           int
	}{
		{"malformed json", `{`, "INVALID_JSON", 400},
		{"unknown field (no image/command injection)", `{"name":"x","epochs":1,"image":"evil","command":"rm -rf /"}`, "INVALID_JSON", 400},
		{"trailing data", `{"name":"x","epochs":1}{"a":1}`, "INVALID_JSON", 400},
		{"wrong type", `{"name":"x","epochs":"three"}`, "INVALID_JSON", 400},
		{"bad epochs", `{"name":"x","epochs":0}`, "INVALID_EPOCHS", 400},
		{"missing dataset", `{"name":"x","epochs":1,"dataset_id":"nope"}`, "DATASET_NOT_FOUND", 404},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			code, out := e.json(t, "POST", "/api/v1/jobs", c.body)
			if code != c.status || errCode(out) != c.want {
				t.Fatalf("got %d %v", code, out)
			}
			if msg := out["error"].(map[string]any)["message"].(string); msg == "" {
				t.Fatal("error message must not be empty")
			}
		})
	}
	if code, out := e.json(t, "GET", "/api/v1/jobs/nope", ""); code != 404 || errCode(out) != "JOB_NOT_FOUND" {
		t.Fatalf("%d %v", code, out)
	}
	if code, out := e.json(t, "POST", "/api/v1/jobs/nope/start", ""); code != 404 || errCode(out) != "JOB_NOT_FOUND" {
		t.Fatalf("%d %v", code, out)
	}
}
