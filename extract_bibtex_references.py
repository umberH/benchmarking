import re
import sys

def extract_cite_keys_from_aux(aux_file):
    """Extract all citation keys from a LaTeX .aux file."""
    cite_keys = set()

    with open(aux_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all \citation{...} commands in .aux file
    pattern = r'\\citation\{([^}]+)\}'
    matches = re.findall(pattern, content)

    for match in matches:
        # Handle multiple keys in one citation command
        keys = [k.strip() for k in match.split(',')]
        cite_keys.update(keys)

    return cite_keys

def parse_bib_file(bib_file):
    """Parse a BibTeX file and return a dictionary of entries with source tracking."""
    entries = {}

    with open(bib_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by @ to get individual entries
    # Pattern to match a complete BibTeX entry
    pattern = r'(@\w+\{[^@]+?\n\})'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        # Extract the citation key
        key_match = re.search(r'@\w+\{([^,\s]+)', match)
        if key_match:
            key = key_match.group(1)
            entries[key] = {
                'content': match,
                'source': bib_file
            }

    return entries

def extract_referenced_bibtex(aux_file, bib_files, output_file):
    """Extract only referenced BibTeX entries from multiple .bib files."""

    # Get all citation keys from aux file
    print(f"Scanning {aux_file} for citations...")
    all_cite_keys = extract_cite_keys_from_aux(aux_file)

    print(f"\nFound {len(all_cite_keys)} unique citation keys:")
    for key in sorted(all_cite_keys):
        print(f"  - {key}")

    # Parse all BibTeX files and track duplicates
    all_bib_entries = {}
    duplicates = {}

    for bib_file in bib_files:
        print(f"\nParsing {bib_file}...")
        entries = parse_bib_file(bib_file)
        print(f"  Found {len(entries)} entries")

        # Check for duplicates
        for key, entry_info in entries.items():
            if key in all_bib_entries:
                if key not in duplicates:
                    duplicates[key] = [all_bib_entries[key]['source']]
                duplicates[key].append(entry_info['source'])
            else:
                all_bib_entries[key] = entry_info

    print(f"\nTotal unique BibTeX entries available: {len(all_bib_entries)}")

    if duplicates:
        print(f"\nDuplicate entries found: {len(duplicates)}")
        for key in sorted(duplicates.keys()):
            sources = duplicates[key]
            print(f"  - {key}")
            for src in sources:
                print(f"      in: {src}")

    # Create case-insensitive lookup for BibTeX entries
    case_insensitive_map = {}
    for key in all_bib_entries.keys():
        case_insensitive_map[key.lower()] = key

    # Extract only referenced entries
    referenced_entries = {}
    missing_keys = []
    case_mismatches = {}

    for key in all_cite_keys:
        if key in all_bib_entries:
            referenced_entries[key] = all_bib_entries[key]
        elif key.lower() in case_insensitive_map:
            # Found with case-insensitive match
            actual_key = case_insensitive_map[key.lower()]
            referenced_entries[key] = all_bib_entries[actual_key]
            case_mismatches[key] = actual_key
        else:
            missing_keys.append(key)

    # Check which duplicates are actually used in citations
    used_duplicates = {k: v for k, v in duplicates.items() if k in all_cite_keys}

    if used_duplicates:
        print(f"\nDuplicates in your citations (kept first occurrence): {len(used_duplicates)}")
        for key in sorted(used_duplicates.keys()):
            sources = used_duplicates[key]
            print(f"  - {key} (using from: {sources[0]})")

    if case_mismatches:
        print(f"\nCase mismatches found and corrected: {len(case_mismatches)}")
        for cite_key, bib_key in sorted(case_mismatches.items()):
            print(f"  - Citation: {cite_key} -> BibTeX: {bib_key}")

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% BibTeX entries extracted from referenced citations\n")
        f.write(f"% Total entries: {len(referenced_entries)}\n")
        if used_duplicates:
            f.write(f"% Note: {len(used_duplicates)} duplicate entries were found and deduplicated\n")
        f.write("\n")

        for key in sorted(referenced_entries.keys()):
            f.write(referenced_entries[key]['content'])
            f.write("\n\n")

    print(f"\nSuccessfully extracted {len(referenced_entries)} unique entries to {output_file}")

    if missing_keys:
        print(f"\nWarning: {len(missing_keys)} citation keys not found in .bib files:")
        for key in sorted(missing_keys):
            print(f"  - {key}")

    return len(referenced_entries), len(missing_keys)

if __name__ == "__main__":
    # CONFIGURE THESE PATHS
    aux_file = "test/output.aux"

    bib_files = [
        "test/references (1).bib",
        "test/references_zotero (1).bib",
        "test/sample-base.bib"
    ]

    output_file = "cleaned_references_no_duplicates.bib"

    # Run extraction
    try:
        extract_referenced_bibtex(aux_file, bib_files, output_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease update the file paths in the script.")
        sys.exit(1)
