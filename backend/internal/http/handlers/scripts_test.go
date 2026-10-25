package handlers_test

import (
	"encoding/json"
	"strings"
	"testing"
)

const projectSrc = "import torch.nn as nn\n\ndef build_model():\n    return nn.Linear(4, 3)\n\ndef get_dataset(data_dir):\n    raise NotImplementedError\n"

func (e *env) uploadScript(t *testing.T, name, filename, content string) (int, map[string]any) {
	t.Helper()
	fields := map[string]string{}
	if name != "" {
		fields["name"] = name
	}
	ct, body := multipartBody(t, fields, filename, content)
	code, _, b := e.do(t, "POST", "/api/v1/scripts", ct, body)
	var out map[string]any
	_ = json.Unmarshal(b, &out)
	return code, out
}

func TestScriptLifecycle(t *testing.T) {
	e := newEnv(t, 1<<20)

	code, sc := e.uploadScript(t, "my-cnn", "cnn.py", projectSrc)
	if code != 201 || sc["name"] != "my-cnn" || sc["key"] != sc["id"].(string)+"/project.py" {
		t.Fatalf("%d %v", code, sc)
	}
	id := sc["id"].(string)

	if code, out := e.json(t, "GET", "/api/v1/scripts", ""); code != 200 || len(out["scripts"].([]any)) != 1 {
		t.Fatalf("%d %v", code, out)
	}
	if code, out := e.json(t, "GET", "/api/v1/scripts/"+id, ""); code != 200 || out["id"] != id {
		t.Fatalf("%d %v", code, out)
	}
	code2, _, src := e.do(t, "GET", "/api/v1/scripts/"+id+"/content", "", nil)
	if code2 != 200 || string(src) != projectSrc {
		t.Fatalf("%d %q", code2, src)
	}
	code2, _, tpl := e.do(t, "GET", "/api/v1/scripts/template", "", nil)
	if code2 != 200 || !strings.Contains(string(tpl), "def build_model()") {
		t.Fatalf("template: %d %q", code2, tpl)
	}

	if code, _, _ := e.do(t, "DELETE", "/api/v1/scripts/"+id, "", nil); code != 204 {
		t.Fatalf("delete: %d", code)
	}
	if code, out := e.json(t, "GET", "/api/v1/scripts/"+id, ""); code != 404 || errCode(out) != "SCRIPT_NOT_FOUND" {
		t.Fatalf("%d %v", code, out)
	}
}

func TestScriptUploadErrors(t *testing.T) {
	e := newEnv(t, 1<<20)
	cases := []struct {
		name, file, body, want string
		status                 int
	}{
		{"not .py", "a.txt", projectSrc, "INVALID_SCRIPT_FILE", 400},
		{"missing contract", "a.py", "print(1)\n", "INVALID_SCRIPT", 400},
		{"empty", "a.py", "", "EMPTY_FILE", 400},
		{"too large", "a.py", projectSrc + strings.Repeat("#", 1<<20), "SCRIPT_TOO_LARGE", 413},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			code, out := e.uploadScript(t, "", c.file, c.body)
			if code != c.status || errCode(out) != c.want {
				t.Fatalf("%d %v", code, out)
			}
		})
	}
	ct, body := multipartBody(t, map[string]string{"name": "x"}, "", "")
	if code, _, b := e.do(t, "POST", "/api/v1/scripts", ct, body); code != 400 || !strings.Contains(string(b), "MISSING_FILE") {
		t.Fatalf("%d %s", code, b)
	}
}

func TestCustomJobWithScriptAndDataset(t *testing.T) {
	e := newEnv(t, 1<<20)
	_, sc := e.uploadScript(t, "", "p.py", projectSrc)
	_, ds := e.upload(t, "d", "d.csv", "1,2,3,4,0\n")
	sid, did := sc["id"].(string), ds["id"].(string)

	// Many workers are fine in custom mode: the dataset is not split by object.
	body := `{"name":"mine","epochs":2,"workers":4,"script_id":"` + sid + `","dataset_id":"` + did + `"}`
	code, j := e.json(t, "POST", "/api/v1/jobs", body)
	if code != 201 || j["data_mode"] != "custom" || j["script_id"] != sid || j["dataset_id"] != did || j["workers"].(float64) != 4 {
		t.Fatalf("%d %v", code, j)
	}
	jobID := j["id"].(string)

	if code, out := e.json(t, "DELETE", "/api/v1/scripts/"+sid, ""); code != 409 || errCode(out) != "SCRIPT_IN_USE" {
		t.Fatalf("script must be protected while a job uses it: %d %v", code, out)
	}
	if code, j = e.json(t, "POST", "/api/v1/jobs/"+jobID+"/start", ""); code != 200 || j["status"] != "RUNNING" {
		t.Fatalf("%d %v", code, j)
	}
	if code, _ := e.json(t, "POST", "/api/v1/jobs/"+jobID+"/stop", ""); code != 202 {
		t.Fatalf("stop: %d", code)
	}
	if err := e.jobs.Reconcile(t.Context()); err != nil {
		t.Fatal(err)
	}
	if code, _, _ := e.do(t, "DELETE", "/api/v1/scripts/"+sid, "", nil); code != 204 {
		t.Fatalf("script should be deletable once the job finished: %d", code)
	}
}

func TestCustomJobWithoutDataset(t *testing.T) {
	e := newEnv(t, 1<<20)
	_, sc := e.uploadScript(t, "", "p.py", projectSrc)
	code, j := e.json(t, "POST", "/api/v1/jobs", `{"name":"nodata","epochs":1,"script_id":"`+sc["id"].(string)+`"}`)
	if code != 201 || j["data_mode"] != "custom" || j["dataset_id"] != nil {
		t.Fatalf("%d %v", code, j)
	}
	if code, j = e.json(t, "POST", "/api/v1/jobs/"+j["id"].(string)+"/start", ""); code != 200 || j["status"] != "RUNNING" {
		t.Fatalf("%d %v", code, j)
	}
}

func TestInvalidCustomJobs(t *testing.T) {
	e := newEnv(t, 1<<20)
	_, sc := e.uploadScript(t, "", "p.py", projectSrc)
	sid := sc["id"].(string)
	cases := []struct {
		name, body, want string
		status           int
	}{
		{"unknown script", `{"name":"x","epochs":1,"script_id":"nope"}`, "SCRIPT_NOT_FOUND", 404},
		{"custom mode needs a script", `{"name":"x","epochs":1,"data_mode":"custom"}`, "INVALID_DATA_MODE", 400},
		{"script with another mode", `{"name":"x","epochs":1,"script_id":"` + sid + `","data_mode":"cifar10"}`, "INVALID_DATA_MODE", 400},
		{"unknown dataset", `{"name":"x","epochs":1,"script_id":"` + sid + `","dataset_id":"nope"}`, "DATASET_NOT_FOUND", 404},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			code, out := e.json(t, "POST", "/api/v1/jobs", c.body)
			if code != c.status || errCode(out) != c.want {
				t.Fatalf("got %d %v", code, out)
			}
		})
	}
}
