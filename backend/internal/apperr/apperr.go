// Package apperr defines the error type shared by services and the HTTP
// layer. Services return *Error values; handlers translate Kind into an
// HTTP status and render {"error":{"code","message"}}. The wrapped cause is
// for logs only and is never sent to clients.
package apperr

import "errors"

type Kind int

const (
	KindInvalid Kind = iota + 1
	KindNotFound
	KindConflict
	KindTooLarge
	KindInternal
)

type Error struct {
	Kind    Kind
	Code    string
	Message string
	Err     error
}

func (e *Error) Error() string {
	if e.Err != nil {
		return e.Code + ": " + e.Message + ": " + e.Err.Error()
	}
	return e.Code + ": " + e.Message
}

func (e *Error) Unwrap() error { return e.Err }

func Invalid(code, msg string) *Error  { return &Error{Kind: KindInvalid, Code: code, Message: msg} }
func NotFound(code, msg string) *Error { return &Error{Kind: KindNotFound, Code: code, Message: msg} }
func Conflict(code, msg string) *Error { return &Error{Kind: KindConflict, Code: code, Message: msg} }
func TooLarge(code, msg string) *Error { return &Error{Kind: KindTooLarge, Code: code, Message: msg} }

// Internal wraps an unexpected failure. The message is generic on purpose.
func Internal(err error) *Error {
	return &Error{Kind: KindInternal, Code: "INTERNAL_ERROR", Message: "internal server error", Err: err}
}

// From returns err as an *Error, wrapping unknown errors as Internal.
func From(err error) *Error {
	var ae *Error
	if errors.As(err, &ae) {
		return ae
	}
	return Internal(err)
}
