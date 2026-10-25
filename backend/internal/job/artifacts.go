package job

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"time"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/storage"
)

// Artifact is one checkpoint file a training run saved. train.py writes
// <checkpoint bucket>/<job id>/epoch_N.pt after every epoch; the file with
// the highest N is the trained model.
type Artifact struct {
	Name         string    `json:"name"`
	Epoch        int       `json:"epoch"`
	SizeBytes    int64     `json:"size_bytes"`
	LastModified time.Time `json:"last_modified"`
}

var checkpointRe = regexp.MustCompile(`^epoch_(\d+)\.pt$`)

// SetArtifacts enables listing/downloading checkpoints from the bucket that
// train.py writes to.
func (s *Service) SetArtifacts(store storage.ObjectStorage, bucket string) {
	s.artifacts, s.artifactBucket = store, bucket
}

// Artifacts lists a job's checkpoints, oldest epoch first.
func (s *Service) Artifacts(ctx context.Context, id string) ([]Artifact, error) {
	if _, err := s.Get(ctx, id); err != nil {
		return nil, err
	}
	if s.artifacts == nil {
		return nil, apperr.Invalid("ARTIFACTS_DISABLED", "model download is not enabled on this server")
	}
	objs, err := s.artifacts.List(ctx, s.artifactBucket, id+"/")
	if err != nil {
		return nil, apperr.Internal(err)
	}
	out := []Artifact{}
	for _, o := range objs {
		name := o.Key[len(id)+1:]
		m := checkpointRe.FindStringSubmatch(name)
		if m == nil {
			continue
		}
		epoch, _ := strconv.Atoi(m[1])
		out = append(out, Artifact{Name: name, Epoch: epoch, SizeBytes: o.Size, LastModified: o.LastModified})
	}
	for i := 1; i < len(out); i++ { // insertion sort by epoch; lists are short
		for j := i; j > 0 && out[j].Epoch < out[j-1].Epoch; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out, nil
}

// Model opens a checkpoint for download: the given epoch, or the latest when
// epoch < 0.
func (s *Service) Model(ctx context.Context, id string, epoch int) (Artifact, io.ReadCloser, error) {
	list, err := s.Artifacts(ctx, id)
	if err != nil {
		return Artifact{}, nil, err
	}
	if len(list) == 0 {
		return Artifact{}, nil, apperr.NotFound("NO_CHECKPOINT", "this job has not saved a model yet")
	}
	pick := list[len(list)-1]
	if epoch >= 0 {
		found := false
		for _, a := range list {
			if a.Epoch == epoch {
				pick, found = a, true
			}
		}
		if !found {
			return Artifact{}, nil, apperr.NotFound("NO_CHECKPOINT", fmt.Sprintf("no checkpoint for epoch %d", epoch))
		}
	}
	rc, err := s.artifacts.Get(ctx, s.artifactBucket, id+"/"+pick.Name)
	if errors.Is(err, storage.ErrNotFound) {
		return Artifact{}, nil, apperr.NotFound("NO_CHECKPOINT", "checkpoint file is missing from storage")
	}
	if err != nil {
		return Artifact{}, nil, apperr.Internal(err)
	}
	return pick, rc, nil
}
