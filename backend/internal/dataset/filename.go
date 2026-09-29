package dataset

import (
	"path"
	"strings"
	"unicode"
	"unicode/utf8"

	"dtdl/backend/internal/apperr"
)

const maxFilenameLen = 255

// SanitizeFilename validates a client-supplied filename and returns
// (safe, display). safe is what goes into the object key: only
// [A-Za-z0-9._-]. display is the cleaned original name kept in metadata.
// Storage keys are never derived from anything else the client sends.
func SanitizeFilename(raw string) (safe, display string, err error) {
	if !utf8.ValidString(raw) {
		return "", "", apperr.Invalid("INVALID_FILENAME", "filename is not valid UTF-8")
	}
	for _, r := range raw {
		if unicode.IsControl(r) {
			return "", "", apperr.Invalid("INVALID_FILENAME", "filename contains control characters")
		}
	}
	// Browsers may send a full Windows path; keep only the last element.
	base := path.Base(strings.ReplaceAll(raw, `\`, "/"))
	base = strings.TrimSpace(base)
	if base == "" || base == "." || base == ".." || base == "/" {
		return "", "", apperr.Invalid("INVALID_FILENAME", "filename is required")
	}
	if len(base) > maxFilenameLen {
		return "", "", apperr.Invalid("INVALID_FILENAME", "filename is too long")
	}
	var b strings.Builder
	for _, r := range base {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	safe = strings.TrimLeft(b.String(), ".") // no hidden files / leading dots
	if safe == "" {
		safe = "file"
	}
	return safe, base, nil
}
