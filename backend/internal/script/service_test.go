package script_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/repository"
	"dtdl/backend/internal/script"
	"dtdl/backend/internal/storage"
)

const good = "import torch\n\ndef build_model():\n    return None\n\ndef get_dataset(data_dir):\n    return None\n"

type stubUsage struct{ inUse bool }

func (s stubUsage) ScriptInUse(context.Context, string) (bool, error) { return s.inUse, nil }

func newSvc(t *testing.T, maxBytes int64) (*script.Service, *storage.Memory) {
	t.Helper()
	db, err := repository.Open("sqlite::memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	store := storage.NewMemory()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return script.NewService(repository.NewScriptRepo(db), store, "scripts", maxBytes, log), store
}

func code(err error) string {
	var ae *apperr.Error
	if errors.As(err, &ae) {
		return ae.Code
	}
	return ""
}

func TestValidate(t *testing.T) {
	cases := []struct {
		name, src, want string
	}{
		{"good", good, ""},
		{"assignment form", "build_model = make\nget_dataset = other\n", ""},
		{"import form", "from mylib import build_model, get_dataset\n", ""},
		{"empty", "", "EMPTY_FILE"},
		{"binary", "def build_model(): pass\x00\x01", "INVALID_SCRIPT"},
		{"missing get_dataset", "def build_model():\n    pass\n", "INVALID_SCRIPT"},
		{"missing both", "print('hi')\n", "INVALID_SCRIPT"},
		{"commented out", "# def build_model():\n# def get_dataset(d):\n", "INVALID_SCRIPT"},
		{"indented (method, not top level)", "class A:\n    def build_model(self): pass\n    def get_dataset(self): pass\n", "INVALID_SCRIPT"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := code(script.Validate([]byte(c.src))); got != c.want {
				t.Fatalf("got %q want %q", got, c.want)
			}
		})
	}
}

func TestUploadStoresUnderFixedKey(t *testing.T) {
	svc, store := newSvc(t, 1<<20)
	sc, err := svc.Upload(t.Context(), script.UploadInput{Filename: `C:\evil\..\my project.py`, Body: strings.NewReader(good)})
	if err != nil {
		t.Fatal(err)
	}
	if sc.Key != sc.ID+"/project.py" || sc.Bucket != "scripts" {
		t.Fatalf("key must not depend on the client filename: %+v", sc)
	}
	if sc.OriginalFilename != "my project.py" || sc.Name != "my project.py" {
		t.Fatalf("display name: %+v", sc)
	}
	rc, err := store.Get(t.Context(), sc.Bucket, sc.Key)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(rc)
	if !bytes.Equal(b, []byte(good)) {
		t.Fatalf("stored content differs: %q", b)
	}
}

func TestUploadRejections(t *testing.T) {
	svc, _ := newSvc(t, 100)
	cases := []struct {
		name, file, body, want string
	}{
		{"not python", "notes.txt", good, "INVALID_SCRIPT_FILE"},
		{"too large", "big.py", good + strings.Repeat("#", 200), "SCRIPT_TOO_LARGE"},
		{"empty", "e.py", "", "EMPTY_FILE"},
		{"no contract", "x.py", "print(1)\n", "INVALID_SCRIPT"},
		{"no filename", "", good, "INVALID_FILENAME"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := svc.Upload(t.Context(), script.UploadInput{Filename: c.file, Body: strings.NewReader(c.body)})
			if code(err) != c.want {
				t.Fatalf("got %v want %s", err, c.want)
			}
		})
	}
}

func TestDeleteBlockedWhileInUse(t *testing.T) {
	svc, store := newSvc(t, 1<<20)
	sc, err := svc.Upload(t.Context(), script.UploadInput{Filename: "p.py", Body: strings.NewReader(good)})
	if err != nil {
		t.Fatal(err)
	}
	svc.SetUsageChecker(stubUsage{inUse: true})
	if got := code(svc.Delete(t.Context(), sc.ID)); got != "SCRIPT_IN_USE" {
		t.Fatalf("got %q", got)
	}
	svc.SetUsageChecker(stubUsage{})
	if err := svc.Delete(t.Context(), sc.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get(t.Context(), sc.Bucket, sc.Key); !errors.Is(err, storage.ErrNotFound) {
		t.Fatalf("object should be gone: %v", err)
	}
	if got := code(svc.Delete(t.Context(), sc.ID)); got != "SCRIPT_NOT_FOUND" {
		t.Fatalf("got %q", got)
	}
}

func TestTemplateFollowsOwnContract(t *testing.T) {
	if err := script.Validate([]byte(script.Template)); err != nil {
		t.Fatalf("the served template must pass validation: %v", err)
	}
}
