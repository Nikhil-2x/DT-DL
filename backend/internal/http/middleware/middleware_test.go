package middleware

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRecoverReturnsJSON500WithoutLeaking(t *testing.T) {
	var logs bytes.Buffer
	log := slog.New(slog.NewTextHandler(&logs, nil))
	h := WithRequestID(Recover(log, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("secret-credential-in-panic")
	})))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/x", nil))

	if rec.Code != 500 || !strings.Contains(rec.Body.String(), "INTERNAL_ERROR") {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	if strings.Contains(rec.Body.String(), "secret") {
		t.Fatal("panic value leaked to client")
	}
	if !strings.Contains(logs.String(), "secret-credential-in-panic") {
		t.Fatal("panic should be logged")
	}
}

func TestClientRequestIDIsIgnored(t *testing.T) {
	h := WithRequestID(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set(HeaderRequestID, "attacker\ninjected")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if got := rec.Header().Get(HeaderRequestID); got == "" || strings.Contains(got, "attacker") {
		t.Fatalf("got %q", got)
	}
}
