package tracing

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/callbacks"
)

// Annotation is a human review on a trace.
type Annotation struct {
	ID        string    `json:"id"`
	RunID     string    `json:"run_id"`
	Status    string    `json:"status"`
	Reviewer  string    `json:"reviewer,omitempty"`
	Comment   string    `json:"comment,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// AnnotationQueue manages human review annotations.
type AnnotationQueue struct {
	mu          sync.RWMutex
	annotations map[string][]Annotation
}

// NewAnnotationQueue creates an empty annotation queue.
func NewAnnotationQueue() *AnnotationQueue {
	return &AnnotationQueue{annotations: make(map[string][]Annotation)}
}

// Submit adds a run to the review queue.
func (q *AnnotationQueue) Submit(runID string) *Annotation {
	now := time.Now().UTC()
	a := Annotation{
		ID:        newAnnotationID(),
		RunID:     runID,
		Status:    "pending",
		CreatedAt: now,
		UpdatedAt: now,
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.annotations[runID] = append(q.annotations[runID], a)
	return &a
}

// Review sets the review status for an annotation.
func (q *AnnotationQueue) Review(annotationID, status, reviewer, comment string) error {
	if status != "approved" && status != "rejected" {
		return fmt.Errorf("tracing: invalid review status %q", status)
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	for runID, list := range q.annotations {
		for i, a := range list {
			if a.ID == annotationID {
				a.Status = status
				a.Reviewer = reviewer
				a.Comment = comment
				a.UpdatedAt = time.Now().UTC()
				q.annotations[runID][i] = a
				return nil
			}
		}
	}
	return fmt.Errorf("tracing: annotation %q not found", annotationID)
}

// Pending returns all pending annotations.
func (q *AnnotationQueue) Pending() []Annotation {
	q.mu.RLock()
	defer q.mu.RUnlock()
	var out []Annotation
	for _, list := range q.annotations {
		for _, a := range list {
			if a.Status == "pending" {
				out = append(out, a)
			}
		}
	}
	return out
}

// List returns all annotations for a run.
func (q *AnnotationQueue) List(runID string) []Annotation {
	q.mu.RLock()
	defer q.mu.RUnlock()
	out := make([]Annotation, len(q.annotations[runID]))
	copy(out, q.annotations[runID])
	return out
}

// Stats returns annotation statistics.
func (q *AnnotationQueue) Stats() map[string]int {
	q.mu.RLock()
	defer q.mu.RUnlock()
	stats := map[string]int{"pending": 0, "approved": 0, "rejected": 0}
	for _, list := range q.annotations {
		for _, a := range list {
			stats[a.Status]++
		}
	}
	return stats
}

func newAnnotationID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// AnnotationHandler auto-submits runs to an annotation queue when errors are detected.
type AnnotationHandler struct {
	callbacks.NoOpHandler
	queue *AnnotationQueue
}

// NewAnnotationHandler creates an AnnotationHandler linked to the given queue.
func NewAnnotationHandler(queue *AnnotationQueue) *AnnotationHandler {
	return &AnnotationHandler{queue: queue}
}

func (h *AnnotationHandler) OnError(ctx context.Context, source string, err error) {
	runID := callbacks.RunIDFromContext(ctx)
	if runID == "" {
		return
	}
	h.queue.Submit(runID)
}
