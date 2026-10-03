"""
Main entry point for the Intelligent Document Assistant.
Provides an interactive command-line interface powered by Rich,
supporting multi-turn Q&A, document summarization, deterministic calculations,
external document loading, document pinning, and strict grounding.
"""

import re
import sys
from pathlib import Path

# Configure UTF-8 on Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.assistant import DocumentAssistant
from src.retrieval import default_repository

console = Console(force_terminal=True, legacy_windows=False)


def display_welcome_banner(session_id: str):
    """Renders the startup banner and available commands."""
    welcome_text = (
        "[bold cyan]DocGrok[/bold cyan] (Agentic Document Assistant with LangGraph & LangChain)\n"
        f"[dim]Active Session ID: [yellow]{session_id}[/yellow] | Storage: sessions/ | Logs: logs/assistant.log[/dim]\n\n"
        "[bold]Document Scoping Modes:[/bold]\n"
        " • [bold green]General Corpus Search:[/bold green] Just ask a question; DocGrok retrieves matching documents.\n"
        " • [bold green]Focus on Specific Document:[/bold green] Type [yellow]use <doc_id>[/yellow] or prefix [yellow][<doc_id>] <query>[/yellow].\n"
        " • [bold green]Load External Document:[/bold green] Type [yellow]load <path> [--persist][/yellow] (PDF, Excel, CSV, Images, JSON, Text).\n\n"
        "[bold]Commands:[/bold]\n"
        " [cyan]docs[/cyan] (list corpus) | [cyan]use <doc_id>[/cyan] (restrict) | [cyan]unpin[/cyan] (search all) |\n"
        " [cyan]load <path> [--persist][/cyan] (import) | [cyan]remove <id> [--permanent][/cyan] (delete) |\n"
        " [cyan]sessions[/cyan] (list saved) | [cyan]resume <id>[/cyan] (continue session) | [cyan]new[/cyan] (fresh session) |\n"
        " [cyan]stats[/cyan] | [bold red]exit[/bold red]"
    )
    console.print(Panel(welcome_text, title="DocGrok", border_style="cyan"))


def display_documents_table():
    """Displays indexed documents and their categories."""
    table = Table(title="Indexed Enterprise Documents", border_style="blue")
    table.add_column("Doc ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Category", style="green")
    table.add_column("Key Metric", style="yellow")

    for chunk in default_repository.get_all_documents():
        amt = chunk.metadata.get("amount")
        amt_str = f"${amt:,.2f}" if amt else "N/A"
        table.add_row(chunk.doc_id, chunk.title, chunk.doc_type, amt_str)

    console.print(table)


def display_sessions_table(assistant: DocumentAssistant):
    """Displays all persisted conversation sessions stored under sessions/."""
    sessions = assistant.list_saved_sessions()
    if not sessions:
        console.print("[dim]No saved sessions found under sessions/[/dim]")
        return

    table = Table(title="Persisted Sessions (sessions/)", border_style="magenta")
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Updated (UTC)", style="dim")
    table.add_column("Turns", justify="right", style="green")
    table.add_column("Pinned Doc", style="yellow")
    table.add_column("Active Documents", style="white")
    table.add_column("Summary Preview", style="dim", max_width=40)

    for s in sessions:
        sid = s["session_id"]
        # Highlight active session
        display_sid = f"[bold green]* {sid}[/bold green]" if sid == assistant.session_id else sid
        updated = s.get("updated_at", "")[:19].replace("T", " ")
        turns = str(s.get("turns_count", 0))
        pinned = s.get("pinned_doc_id") or "None"
        active_docs = ", ".join(s.get("active_documents", [])) or "None"
        summary = s.get("conversation_summary", "")
        summary_preview = (summary[:37] + "...") if len(summary) > 40 else (summary or "No summary yet")

        table.add_row(display_sid, updated, turns, pinned, active_docs, summary_preview)

    console.print(table)


def run_cli():
    """Main interactive loop."""
    import argparse

    parser = argparse.ArgumentParser(description="DocGrok: Intelligent Document Assistant CLI")
    parser.add_argument("--session", "-s", type=str, default=None, help="Resume or start a specific session ID")
    cli_args, _ = parser.parse_known_args()

    try:
        assistant = DocumentAssistant(session_id=cli_args.session)
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        console.print("[yellow]Please ensure your GEMINI_API_KEY is configured in .env[/yellow]")
        sys.exit(1)

    display_welcome_banner(assistant.session_id)
    if cli_args.session and assistant.session_manager.session_exists(cli_args.session):
        console.print(f"[bold green]Successfully resumed session '{cli_args.session}' from disk.[/bold green]")
        if assistant.conversation_summary:
            console.print(Panel(assistant.conversation_summary, title="Prior Conversation Context", border_style="dim cyan"))

    while True:
        try:
            # Show current session and scope in the prompt
            session_label = f"[[dim]Session:[/dim] [cyan]{assistant.session_id}[/cyan]]"
            if assistant.pinned_doc_ids:
                docs_label = ", ".join(assistant.pinned_doc_ids)
                scope_label = f"[Scope: [bold yellow]{docs_label}[/bold yellow]]"
            else:
                scope_label = "[Scope: [dim]Corpus Search[/dim]]"

            console.print()
            user_input = console.input(f"[bold blue]You[/bold blue] {session_label} {scope_label} > ").strip()

            if not user_input:
                continue

            cmd_lower = user_input.lower()

            # Global exit commands
            if cmd_lower in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            # List persisted sessions
            elif cmd_lower in ("sessions", "history"):
                display_sessions_table(assistant)
                continue

            # Resume an existing session from disk
            elif cmd_lower.startswith("resume ") or cmd_lower.startswith("switch "):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    target_sid = parts[1].strip()
                    if assistant.resume_session(target_sid):
                        console.print(f"[bold green]Resumed session:[/bold green] [yellow]{target_sid}[/yellow]")
                        if assistant.pinned_doc_id:
                            console.print(f"[dim]Restored document focus:[/dim] [yellow]{assistant.pinned_doc_id}[/yellow]")
                        if assistant.conversation_summary:
                            console.print(Panel(assistant.conversation_summary, title="Restored Context Summary", border_style="dim cyan"))
                    else:
                        console.print(f"[bold red]Session '{target_sid}' not found.[/bold red] Type [cyan]sessions[/cyan] to see available sessions.")
                continue

            # Start a new session
            elif cmd_lower.startswith("new") or cmd_lower == "reset":
                parts = user_input.split(maxsplit=1)
                requested_sid = parts[1].strip() if len(parts) > 1 and not cmd_lower.startswith("reset") else None
                new_sid = assistant.reset_session(requested_sid)
                console.print(f"[green]Started new session:[/green] [yellow]{new_sid}[/yellow]")
                continue

            # List documents
            elif cmd_lower in ("docs", "list"):
                display_documents_table()
                continue

            # Stats command
            elif cmd_lower == "stats":
                stats = assistant.repo.get_statistics()
                corpus_dir = stats.get("corpus_directory", "document_corpus")
                tot_docs = stats.get("total_documents", 0)
                tot_vol = stats.get("total_dollar_volume", 0.0)
                avg_amt = stats.get("average_amount", 0.0)
                doc_types = stats.get("document_types", {})

                type_breakdown = "\n".join([f" • [cyan]{k}:[/cyan] {v}" for k, v in doc_types.items()])

                stats_text = (
                    f"[bold]Corpus Directory:[/bold] [yellow]{corpus_dir}[/yellow]\n"
                    f"[bold]Total Documents Indexed:[/bold] {tot_docs} "
                    f"([dim]{stats.get('documents_with_amounts', 0)} with financial values[/dim])\n"
                    f"[bold]Total Dollar Volume:[/bold] [bold green]${tot_vol:,.2f}[/bold green]\n"
                    f"[bold]Average Metric Value:[/bold] [green]${avg_amt:,.2f}[/green]\n\n"
                    f"[bold]Category Distribution:[/bold]\n{type_breakdown}"
                )
                console.print(Panel(stats_text, title="Corpus Statistics", border_style="green"))
                continue

            # Unpin document scope
            elif cmd_lower in ("unpin", "clear-doc", "all"):
                assistant.unpin_document()
                console.print("[green]Cleared pinned document scope. Now searching full corpus.[/green]")
                continue

            # Pin one or multiple existing documents
            elif cmd_lower.startswith("use ") or cmd_lower.startswith("doc "):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    raw_ids = [i.strip() for i in re.split(r"[\s,]+", parts[1]) if i.strip()]
                    pinned = assistant.pin_documents(raw_ids)
                    if pinned:
                        console.print(f"[bold green]Focused on {len(pinned)} document(s):[/bold green] [yellow]{', '.join(pinned)}[/yellow]. All queries will now strictly use these documents.")
                    else:
                        console.print(f"[bold red]None of the specified document IDs were found.[/bold red] Type [cyan]docs[/cyan] to see valid IDs.")
                continue

            # Ingest external file(s)
            elif cmd_lower.startswith("load ") or cmd_lower.startswith("import "):
                args = user_input.split()
                if len(args) < 2:
                    console.print("[yellow]Usage: load <filepath1> [filepath2...] [--persist][/yellow]")
                    continue

                persist = "--persist" in args or "-p" in args
                file_paths = [a for a in args[1:] if a not in ("--persist", "-p")]

                loaded_ids = []
                for fp in file_paths:
                    try:
                        chunk = assistant.load_external_document(file_path=fp, persist=persist)
                        loaded_ids.append(chunk.doc_id)
                        persist_msg = " [bold green](persisted to corpus)[/bold green]" if persist else " [dim](in-memory)[/dim]"
                        console.print(f"[bold green]Loaded document:[/bold green] [yellow]{chunk.doc_id}[/yellow] - '{chunk.title}'{persist_msg}")
                    except Exception as ex:
                        console.print(f"[bold red]Failed to load file '{fp}':[/bold red] {ex}")

                if loaded_ids:
                    for lid in loaded_ids:
                        if lid not in assistant.pinned_doc_ids:
                            assistant.pinned_doc_ids.append(lid)
                    console.print(f"[cyan]Active scope focused on: [yellow]{', '.join(assistant.pinned_doc_ids)}[/yellow]. Type [yellow]unpin[/yellow] to return to full corpus search.[/cyan]")
                continue

            # Remove or unload a document from corpus or active memory
            elif cmd_lower.startswith("remove ") or cmd_lower.startswith("delete ") or cmd_lower.startswith("unload "):
                args = user_input.split()
                if len(args) < 2:
                    console.print("[yellow]Usage: remove <doc_id> [--permanent|-p] [--force|-f][/yellow] or [yellow]unload <doc_id>[/yellow]")
                    continue

                target_id = args[1].strip()
                is_unload = cmd_lower.startswith("unload")
                permanent = ("--permanent" in args or "-p" in args) and not is_unload
                force = "--force" in args or "-f" in args

                success, msg = assistant.remove_document(doc_id=target_id, permanent=permanent, force=force)
                if success:
                    console.print(f"[bold green]{msg}[/bold green]")
                else:
                    console.print(f"[bold red]Removal Failed:[/bold red] {msg}")
                continue

            # Inline document tag parsing: e.g. "[FIN-2024-Q1] What was revenue?" or "doc:MED-CLM-7701 summarize"
            inline_match = re.match(r"^\[([a-zA-Z0-9_\-]+)\]\s*(.+)$", user_input) or re.match(r"^doc:([a-zA-Z0-9_\-]+)\s*(.+)$", user_input, re.IGNORECASE)
            turn_target_doc_id = None
            query_to_send = user_input

            if inline_match:
                turn_target_doc_id = inline_match.group(1).upper()
                query_to_send = inline_match.group(2).strip()
                console.print(f"[dim]Turn scope restricted to: {turn_target_doc_id}[/dim]")

            # Execute turn through LangGraph
            with console.status("[bold cyan]Processing through LangGraph workflow...[/bold cyan]", spinner="dots"):
                result_state = assistant.process_message(
                    user_input=query_to_send,
                    target_doc_id=turn_target_doc_id
                )

            formatted_response = assistant.format_output(result_state)
            console.print()
            console.print(Panel(formatted_response, title="Agent Response", border_style="green"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting...[/yellow]")
            break
        except Exception as ex:
            console.print(f"[bold red]Execution Error:[/bold red] {ex}")


if __name__ == "__main__":
    run_cli()
