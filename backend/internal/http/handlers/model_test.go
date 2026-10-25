package handlers_test

import (
	"strings"
	"testing"
)

func TestModelDownload(t *testing.T) {
	e := newEnv(t, 1<<20)
	code, j := e.json(t, "POST", "/api/v1/jobs", `{"name":"m","epochs":3}`)
	if code != 201 {
		t.Fatalf("%d %v", code, j)
	}
	id := j["id"].(string)

	// No checkpoints yet.
	if code, out := e.json(t, "GET", "/api/v1/jobs/"+id+"/model", ""); code != 404 || errCode(out) != "NO_CHECKPOINT" {
		t.Fatalf("%d %v", code, out)
	}
	if code, out := e.json(t, "GET", "/api/v1/jobs/"+id+"/artifacts", ""); code != 200 || len(out["artifacts"].([]any)) != 0 {
		t.Fatalf("%d %v", code, out)
	}

	// train.py writes <id>/epoch_N.pt; epoch 10 must sort after epoch 2.
	for _, name := range []string{"epoch_0.pt", "epoch_2.pt", "epoch_10.pt", "notes.txt"} {
		if _, err := e.store.Put(t.Context(), "checkpoints", id+"/"+name, strings.NewReader("weights-"+name), -1, ""); err != nil {
			t.Fatal(err)
		}
	}
	// Another job's checkpoints must never leak in.
	_, _ = e.store.Put(t.Context(), "checkpoints", id+"x/epoch_99.pt", strings.NewReader("other"), -1, "")

	code, out := e.json(t, "GET", "/api/v1/jobs/"+id+"/artifacts", "")
	arts := out["artifacts"].([]any)
	if code != 200 || len(arts) != 3 || out["latest_epoch"].(float64) != 10 {
		t.Fatalf("%d %v", code, out)
	}
	if arts[0].(map[string]any)["epoch"].(float64) != 0 || arts[2].(map[string]any)["epoch"].(float64) != 10 {
		t.Fatalf("not sorted by epoch: %v", arts)
	}

	code, hdr, body := e.do(t, "GET", "/api/v1/jobs/"+id+"/model", "", nil)
	if code != 200 || string(body) != "weights-epoch_10.pt" {
		t.Fatalf("latest: %d %q", code, body)
	}
	if cd := hdr.Get("Content-Disposition"); cd != `attachment; filename="`+id+`-epoch_10.pt"` {
		t.Fatalf("Content-Disposition: %q", cd)
	}
	if _, _, body = e.do(t, "GET", "/api/v1/jobs/"+id+"/model?epoch=2", "", nil); string(body) != "weights-epoch_2.pt" {
		t.Fatalf("epoch=2: %q", body)
	}
	if code, out := e.json(t, "GET", "/api/v1/jobs/"+id+"/model?epoch=7", ""); code != 404 || errCode(out) != "NO_CHECKPOINT" {
		t.Fatalf("%d %v", code, out)
	}
	if code, out := e.json(t, "GET", "/api/v1/jobs/"+id+"/model?epoch=abc", ""); code != 400 || errCode(out) != "INVALID_EPOCH" {
		t.Fatalf("%d %v", code, out)
	}
	if code, out := e.json(t, "GET", "/api/v1/jobs/nope/model", ""); code != 404 || errCode(out) != "JOB_NOT_FOUND" {
		t.Fatalf("%d %v", code, out)
	}
}
