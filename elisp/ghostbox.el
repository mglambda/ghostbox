;; -*- lexical-binding: t -*-

(defvar ghostbox-file-path "/home/marius/prog/ai/ghostbox/scripts/ghostbox"
  "Path to the program used by `run-ghostbox'")

(defvar ghostbox--multiline-delimiter "<*>GHOSTBOX_END<*>")
(defvar ghostbox--multiline-delimiter-newlines (concat "\n" ghostbox--multiline-delimiter "\n"))

(defvar ghostbox-arguments `(
							 ,(concat "--multiline_delimiter='" ghostbox--multiline-delimiter "'"))
  "Commandline arguments to pass to `ghostbox'.")

(defvar ghostbox-mode-map
  (let ((map (nconc (make-sparse-keymap) comint-mode-map)))
    ;; example definition
    (define-key map "\t" 'completion-at-point)
    map)
  "Basic mode map for `run-ghostbox'.")

(defvar ghostbox-prompt-regexp "^\\(?:\\[[^@]+@[^@]+\\]\\)"
  "Prompt for `run-ghostbox'.")

(defvar ghostbox-buffer-name "*ghostbox*"
  "Name of the buffer to use for the `run-ghostbox' comint instance.")

(defun run-ghostbox ()
  "Run an inferior instance of `ghostbox' inside Emacs."
  (interactive)
  (let* ((ghostbox-program ghostbox-file-path)
         (buffer (get-buffer-create ghostbox-buffer-name))
         (proc-alive (comint-check-proc buffer))
         (process (get-buffer-process buffer)))
    ;; if the process is dead then re-create the process and reset the
    ;; mode.
    (unless proc-alive
      (with-current-buffer buffer
        (apply 'make-comint-in-buffer "Ghostbox" buffer
               ghostbox-program nil ghostbox-arguments)
        (ghostbox-mode)))
    ;; Regardless, provided we have a valid buffer, we pop to it.
    (when buffer
      (pop-to-buffer buffer))))

(defun ghostbox--initialize ()
  "Helper function to initialize Ghostbox."
  (set (make-local-variable 'comint-process-echoes) t)
  (set (make-local-variable 'comint-use-prompt-regexp) nil)
  (set (make-local-variable 'comint-input-sender) 'ghostbox--input-sender))

(define-derived-mode ghostbox-mode comint-mode "Ghostbox"
  "Major mode for `run-ghostbox'.

\\<ghostbox-mode-map>"
  ;; this sets up the prompt so it matches things like: [foo@bar]
  (setq comint-prompt-regexp ghostbox-prompt-regexp)
  ;; this makes it read only; a contentious subject as some prefer the
  ;; buffer to be overwritable.
  (setq comint-prompt-read-only t)
  ;; this makes it so commands like M-{ and M-} work.
  (set (make-local-variable 'paragraph-separate) "\\'")
  (set (make-local-variable 'font-lock-defaults) '(ghostbox-font-lock-keywords t))
  (set (make-local-variable 'paragraph-start) ghostbox-prompt-regexp))

(add-hook 'ghostbox-mode-hook 'ghostbox--initialize)

(defconst ghostbox-keywords
  '("assume" "connect" "consistencylevel" "count" "create column family"
    "create keyspace" "del" "decr" "describe cluster" "describe"
    "drop column family" "drop keyspace" "drop index" "get" "incr" "list"
    "set" "show api version" "show cluster name" "show keyspaces"
    "show schema" "truncate" "update column family" "update keyspace" "use")
  "List of keywords to highlight in `ghostbox-font-lock-keywords'.")

(defvar ghostbox-font-lock-keywords
  (list
   ;; highlight all the reserved commands.
   `(,(concat "\\_<" (regexp-opt ghostbox-keywords) "\\_>") . font-lock-keyword-face))
  "Additional expressions to highlight in `ghostbox-mode'.")


;; Just listing some comint hooks to help myself
;; taken from https://www.masteringemacs.org/article/comint-writing-command-interpreter
;; comint-dynamic-complete-functions
;;     List of functions called to perform completion.
;; comint-input-filter-functions
;;     Abnormal hook run before input is sent to the process.
;; comint-output-filter-functions
;;     Functions to call after output is inserted into the buffer.
;; comint-preoutput-filter-functions
;;     List of functions to call before inserting Comint output into the buffer.
;; comint-redirect-filter-functions
;;     List of functions to call before inserting redirected process output.
;; comint-redirect-original-filter-function
;;     The process filter that was in place when redirection is started
;; completion-at-point-functions
;;     This is the preferred method for building completion functions in Emacs.
;; Another useful variable is comint-input-sender, which lets you alter the input string mid-stream. Annoyingly its name is inconsistent with the filter functions above.


(provide 'ghostbox)



(defun ghostbox-send-region (begin end)
  "Send contents of region to ghostbox for completion."
  (interactive "r")
  (process-send-region ghostbox-buffer-name begin end))

(defun ghostbox-send-raw (w)
"Send a string to the ghostbox process, without formatting or delimiter.\nSince this function does not append the multiline_delimiter, but ghostbox is run in multiline mode from emacs by default, sending the string may not trigger ghostbox to send a request to the backend. You can still use this to slowly build a longer input to ghostbox, or send complex command etc. When you are done, simply send ghostbox--multiline-delimiter-newlines ."
  (interactive "MMessage")
  (process-send-string ghostbox-buffer-name w))

(defun ghostbox-send-string (w)
  "Send a string to the ghostbox process. Automatically appends newline."
  (interactive "MMessage")
  (process-send-string ghostbox-buffer-name (concat w ghostbox--multiline-delimiter-newlines)))
  
(defun ghostbox-send-buffer (buffer)
  "Send contents of a buffer to ghostbox."
  (interactive "bBuffer:")
  (with-current-buffer buffer
	(process-send-string ghostbox-buffer-name
						 (concat (buffer-string) ghostbox--multiline-delimiter-newlines))))

(defun ghostbox-send-current-buffer ()
  "Sends contents of the current buffer to ghostbox."
  (interactive)
  (process-send-string ghostbox-buffer-name
					   (concat (buffer-string) ghostbox--multiline-delimiter-newlines)))

(defun ghostbox--input-sender (proc w)
  "Adds the delimiter after the string. Intended to be used as a hookeor replacement -function for comint-input-sender, since we need to add delimiter after the user hits enter."
  (process-send-string proc
					   (concat w ghostbox--multiline-delimiter-newlines)))
