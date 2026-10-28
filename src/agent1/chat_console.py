"""Minimal notebook/terminal text interface around ChatSession."""
from agent1.session import UnsavedTurnError


def run_chat(session, input_fn=input, output_fn=print):
    if session.agent is None:
        session.new()
    output_fn('Conversation: ' + session.conversation_id)
    output_fn('/new /list /open ID /history /retry-save /reload /quit')
    while True:
        try:
            text = input_fn('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            output_fn('Chat stopped. Only successfully saved turns are durable.')
            return
        if not text:
            continue
        try:
            if text == '/quit':
                output_fn('Chat stopped. Only successfully saved turns are durable.')
                return
            if text == '/new':
                output_fn('Conversation: ' + session.new())
            elif text == '/list':
                for item in session.list_conversations():
                    output_fn(f"{item['conversation_id']} | {item['updated_at']} | {item['preview'] or '(empty)'}")
            elif text.startswith('/open '):
                output_fn('Opened: ' + session.open(text[6:].strip()))
                for m in session.history():
                    if m['role'] in ('user','assistant'):
                        output_fn(f"{m['role']}: {m['content']}")
            elif text == '/history':
                for m in session.history():
                    if m['role'] in ('user','assistant'):
                        output_fn(f"{m['role']}: {m['content']}")
            elif text == '/retry-save':
                session.retry_save()
                output_fn('The completed turn is now saved; it was not executed again.')
            elif text == '/reload':
                if input_fn('Discard unsaved progress and reload? Type yes: ').strip().lower() == 'yes':
                    session.reload_saved()
                    output_fn('Reloaded the last saved checkpoint.')
            elif text.startswith('/'):
                output_fn('Unknown command. Use /new /list /open ID /history /retry-save /reload /quit.')
            else:
                output_fn('Agent1: ' + session.send(text))
        except UnsavedTurnError as exc:
            output_fn('Agent1 (NOT CONFIRMED SAVED): ' + exc.answer)
            output_fn('Saving failed. Use /retry-save rather than sending the message again.')
        except Exception as exc:
            # Avoid printing connection details/credentials in the chat surface.
            output_fn(f'Operation failed ({type(exc).__name__}). No automatic retry was made. '
                      'After a processing failure, /reload discards the unsaved turn.')
