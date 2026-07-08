package graph

import (
	"context"
	"testing"
)

func TestFileCheckpointerRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cp, err := NewFileCheckpointer[map[string]int](dir)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Save two checkpoints for a thread.
	for i, state := range []map[string]int{{"n": 1}, {"n": 2}} {
		if err := cp.Save(ctx, "t1", Checkpoint[map[string]int]{ThreadID: "t1", State: state, StepCount: i}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := cp.Load(ctx, "t1")
	if err != nil || got == nil {
		t.Fatalf("load: %v %v", got, err)
	}
	if got.State["n"] != 2 {
		t.Fatalf("expected latest state {n:2}, got %#v", got.State)
	}

	list, err := cp.List(ctx, "t1")
	if err != nil || len(list) != 2 {
		t.Fatalf("list: %d %v", len(list), err)
	}

	if err := cp.Delete(ctx, "t1"); err != nil {
		t.Fatal(err)
	}
	if got, _ := cp.Load(ctx, "t1"); got != nil {
		t.Fatalf("expected nil after delete, got %#v", got)
	}
}

func TestSanitizeThreadID(t *testing.T) {
	if s := sanitizeThreadID("../etc/passwd"); s == "../etc/passwd" {
		t.Fatalf("dangerous id not sanitized: %q", s)
	}
	if s := sanitizeThreadID(""); s != "default" {
		t.Fatalf("empty id: %q", s)
	}
}
