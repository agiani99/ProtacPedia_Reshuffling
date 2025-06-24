# ProtacPedia_Reshuffling

## Short Description

Key Features:
Automatic Column Detection: The script now automatically detects and maps the requested columns from the input CSV, handling case variations:
"Ligand PDB" or "PDB Ligand" → Ligand_PDB
"Ligand ID" → Ligand_ID
"Pubmed" → Pubmed
"Target" → Target

Output Files:
Main output: protac_fragments_improved_alt_with_metadata.csv (successful fragmentations only)
Debug output: protac_fragments_improved_alt_debug_with_metadata.csv (all results including failures)

Flexible Column Handling: The script will automatically include any of the requested columns that exist in your input file, and skip any that don't exist.
Informative Console Output: The script will tell you:

Core fragmentation columns: PROTACDB ID, Dc50, E3_Binder_Fragment, Warhead_Fragment, Linker_Fragment, Atom_Replacements
Additional metadata columns: Ligand_PDB, Ligand_ID, Pubmed, Target (whichever are available in your input)

The script maintains all the advanced features from before: atom replacements, detailed error tracking, multiple matching strategies, while now preserving the important metadata you need for downstream analysis.
Just run this script with your protacdb_20220210.csv file and it will automatically detect and include the metadata columns in both output files!RetryClaude can make mistakes. Please double-check responses.
