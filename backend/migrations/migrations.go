// Package migrations embeds the SQL schema migrations, applied in filename
// order by repository.Open.
package migrations

import "embed"

//go:embed *.sql
var FS embed.FS
