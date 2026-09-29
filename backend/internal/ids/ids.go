// Package ids generates opaque random identifiers. IDs are lowercase hex, so
// they are safe to use in object keys, container names and CLI arguments.
package ids

import (
	"crypto/rand"
	"encoding/hex"
)

func New() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		panic("ids: crypto/rand failed: " + err.Error())
	}
	return hex.EncodeToString(b)
}
