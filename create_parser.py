import argparse

def create_parser():
    """Initialize and return the argument parser with all commands."""
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")

    # Define parent parsers for shared arguments
    path_arg_parent = argparse.ArgumentParser(add_help=False)
    path_arg_parent.add_argument(
        "-p",
        "--path",
        type=str,
        required=False,
        help="Path to a directory containing documents to index.",
    )

    eval_file_arg_parent = argparse.ArgumentParser(add_help=False)
    eval_file_arg_parent.add_argument(
        "-f",
        "--eval_file",
        type=str,
        required=False,
        help="Path to a .json file with question/expected_answer pairs.",
    )
    
    # ← NOVO: Parent para retrievers (funciona com subparsers)
    retriever_parent = argparse.ArgumentParser(add_help=False)
    retriever_parent.add_argument("--original", action="store_true", help="Retriever original (top_k*3)")
    retriever_parent.add_argument("--refatorado", action="store_true", help="Retriever refatorado (top_k=5)")

    # Add global arguments to main parser (backup)
    parser.add_argument("-p", "--path", type=str, required=False, help="Path to documents.")
    parser.add_argument("-f", "--eval_file", type=str, required=False, help="Eval JSON file.")

    # Subparsers COM retriever_parent
    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)

    subparsers.add_parser(
        "run",
        help="Run full pipeline: reset, add, evaluate.",
        parents=[path_arg_parent, eval_file_arg_parent, retriever_parent],  # ← ADICIONADO
    )
    subparsers.add_parser("reset", help="Reset the database")
    subparsers.add_parser(
        "add", 
        help="Add documents.", 
        parents=[path_arg_parent, retriever_parent]  # ← ADICIONADO
    )
    subparsers.add_parser(
        "evaluate", 
        help="Evaluate the model", 
        parents=[eval_file_arg_parent, retriever_parent]  # ← ADICIONADO
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the documents")
    query_parser.add_argument("prompt", type=str, help="What to search for.")
    query_parser.add_argument("--original", action="store_true", help="Retriever original")  # ← DIRETO
    query_parser.add_argument("--refatorado", action="store_true", help="Retriever refatorado")  # ← DIRETO

    return parser
