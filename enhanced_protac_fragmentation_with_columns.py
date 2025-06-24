import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import warnings

def add_hydrogens_final(mol):
    """Add explicit hydrogens to molecule at the end of processing"""
    if mol is None:
        return None, "Input molecule is None"
    try:
        return Chem.AddHs(mol), "Success"
    except Exception as e:
        return mol, f"AddHs failed: {str(e)}"

def canonicalize_tautomers(mol):
    """Convert molecule to canonical tautomer form"""
    if mol is None:
        return None, "Input molecule is None"
    try:
        # Use RDKit's tautomer canonicalizer
        tautomer_canon = rdMolStandardize.TautomerCanonicalizer()
        canonical_mol = tautomer_canon.Canonicalize(mol)
        return canonical_mol, "Success"
    except Exception as e:
        return mol, f"Tautomer canonicalization failed: {str(e)}"

def normalize_molecule(mol):
    """Comprehensive molecule normalization including tautomers"""
    if mol is None:
        return None, "Input molecule is None"
    try:
        # Step 1: Basic cleanup
        mol = Chem.RemoveHs(mol)
        
        # Step 2: Neutralize charges if possible
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        # Step 3: Canonicalize tautomers
        mol, tautomer_error = canonicalize_tautomers(mol)
        
        # Step 4: Final sanitization
        Chem.SanitizeMol(mol)
        
        return mol, "Success"
    except Exception as e:
        return mol, f"Normalization failed: {str(e)}"

def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form"""
    if not smiles or pd.isna(smiles):
        return None, "Empty or NaN SMILES"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "MolFromSmiles returned None - invalid SMILES"
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), "Success"
    except Exception as e:
        return None, f"SMILES canonicalization failed: {str(e)}"

def remove_stereochemistry(mol):
    """Remove stereochemistry from molecule for better matching"""
    if mol is None:
        return None, "Input molecule is None"
    try:
        mol_copy = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol_copy)
        return mol_copy, "Success"
    except Exception as e:
        return mol, f"Stereochemistry removal failed: {str(e)}"

def try_alternative_representations(mol):
    """Try different molecular representations for better matching"""
    representations = [mol]  # Start with original
    
    try:
        # Try without hydrogens
        no_h = Chem.RemoveHs(mol)
        if no_h:
            representations.append(no_h)
    except:
        pass
    
    try:
        # Try without stereochemistry
        no_stereo, _ = remove_stereochemistry(mol)
        if no_stereo:
            representations.append(no_stereo)
    except:
        pass
    
    try:
        # Try normalized
        normalized, _ = normalize_molecule(mol)
        if normalized:
            representations.append(normalized)
    except:
        pass
    
    return representations

def normalize_molecule_simple(mol):
    """Simple normalization without detailed error tracking"""
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except:
        return mol

def generate_atom_replacement_variants(mol, replacements=None):
    """
    Generate molecular variants with atom replacements (bioisosteres)
    Default replacements: O->N, O->C, N->O, N->C, C->O, C->N
    Returns list of (modified_mol, replacement_description) tuples
    """
    if mol is None:
        return []
    
    if replacements is None:
        # Common bioisosteric replacements for sp3 atoms
        replacements = [
            (8, 7),   # O -> N
            (8, 6),   # O -> C  
            (7, 8),   # N -> O
            (7, 6),   # N -> C
            (6, 8),   # C -> O
            (6, 7),   # C -> N
        ]
    
    variants = []
    
    for from_atomic_num, to_atomic_num in replacements:
        # Find all atoms of the source type that are sp3 (not in aromatic rings or double bonds)
        candidate_atoms = []
        
        for atom in mol.GetAtoms():
            if (atom.GetAtomicNum() == from_atomic_num and 
                atom.GetHybridization() == Chem.HybridizationType.SP3 and
                not atom.GetIsAromatic()):
                
                # Check if atom is not involved in double/triple bonds
                is_sp3_like = True
                for bond in atom.GetBonds():
                    if bond.GetBondType() not in [Chem.BondType.SINGLE]:
                        is_sp3_like = False
                        break
                
                if is_sp3_like:
                    candidate_atoms.append(atom.GetIdx())
        
        # Generate variants by replacing each candidate atom
        for atom_idx in candidate_atoms:
            try:
                # Create a copy and modify the atom
                mol_copy = Chem.RWMol(mol)
                atom = mol_copy.GetAtomWithIdx(atom_idx)
                
                # Store original formal charge and other properties
                original_formal_charge = atom.GetFormalCharge()
                
                # Change atomic number
                atom.SetAtomicNum(to_atomic_num)
                
                # Adjust formal charge if needed (basic heuristic)
                if from_atomic_num == 8 and to_atomic_num == 7:  # O->N
                    # Oxygen typically neutral, nitrogen might need +1 if same connectivity
                    if original_formal_charge == 0:
                        atom.SetFormalCharge(1)
                elif from_atomic_num == 7 and to_atomic_num == 8:  # N->O
                    # Nitrogen +1 -> Oxygen 0
                    if original_formal_charge == 1:
                        atom.SetFormalCharge(0)
                
                # Try to sanitize the molecule
                try:
                    Chem.SanitizeMol(mol_copy)
                    new_mol = mol_copy.GetMol()
                    
                    # Create description
                    atom_symbols = {6: 'C', 7: 'N', 8: 'O'}
                    from_symbol = atom_symbols.get(from_atomic_num, str(from_atomic_num))
                    to_symbol = atom_symbols.get(to_atomic_num, str(to_atomic_num))
                    description = f"{from_symbol}->{to_symbol} at position {atom_idx}"
                    
                    variants.append((new_mol, description))
                    
                except Exception as e:
                    # Skip variants that can't be sanitized
                    continue
                    
            except Exception as e:
                # Skip if atom modification fails
                continue
    
    return variants

def find_substructure_fast_original(parent_mol, substructure_mol):
    """
    Original fast substructure matching with multiple representation strategies
    Returns (match_result, error_details)
    """
    if parent_mol is None or substructure_mol is None:
        return None, "One or both input molecules are None"
    
    error_log = []
    
    # Strategy 1: Direct substructure matching
    try:
        matches = parent_mol.GetSubstructMatches(substructure_mol)
        if matches:
            return matches[0], "Success - Direct match"
        else:
            error_log.append("Strategy 1: Direct GetSubstructMatches - no matches found")
    except Exception as e:
        error_log.append(f"Strategy 1: Direct GetSubstructMatches failed - {str(e)}")
    
    # Strategy 2: Try different representations of both molecules
    try:
        parent_representations = try_alternative_representations(parent_mol)
        substructure_representations = try_alternative_representations(substructure_mol)
        
        for i, parent_rep in enumerate(parent_representations):
            for j, sub_rep in enumerate(substructure_representations):
                if parent_rep is not None and sub_rep is not None:
                    try:
                        matches = parent_rep.GetSubstructMatches(sub_rep)
                        if matches:
                            return matches[0], f"Success - Alternative representations (parent:{i}, sub:{j})"
                    except Exception as e:
                        error_log.append(f"Strategy 2: Alternative rep {i},{j} failed - {str(e)}")
        
        error_log.append("Strategy 2: All alternative representations failed to match")
    except Exception as e:
        error_log.append(f"Strategy 2: Alternative representations setup failed - {str(e)}")
    
    # Strategy 3: Try with simple normalization
    try:
        parent_normalized = normalize_molecule_simple(parent_mol)
        substructure_normalized = normalize_molecule_simple(substructure_mol)
        
        if parent_normalized and substructure_normalized:
            matches = parent_normalized.GetSubstructMatches(substructure_normalized)
            if matches:
                return matches[0], "Success - Normalized molecules"
            else:
                error_log.append("Strategy 3: Normalized molecules - no matches found")
        else:
            error_log.append("Strategy 3: Normalization failed on one or both molecules")
    except Exception as e:
        error_log.append(f"Strategy 3: Normalized matching failed - {str(e)}")
    
    # Strategy 4: Try normalized + no stereochemistry
    try:
        parent_norm_no_stereo = remove_stereochemistry(normalize_molecule_simple(parent_mol))[0]
        substructure_norm_no_stereo = remove_stereochemistry(normalize_molecule_simple(substructure_mol))[0]
        
        if parent_norm_no_stereo and substructure_norm_no_stereo:
            matches = parent_norm_no_stereo.GetSubstructMatches(substructure_norm_no_stereo)
            if matches:
                return matches[0], "Success - Normalized + no stereochemistry"
            else:
                error_log.append("Strategy 4: Normalized + no stereochemistry - no matches found")
        else:
            error_log.append("Strategy 4: Normalization + stereochemistry removal failed")
    except Exception as e:
        error_log.append(f"Strategy 4: Normalized + no stereochemistry failed - {str(e)}")
    
    return None, " | ".join(error_log)

def find_substructure_with_atom_replacements(parent_mol, substructure_mol):
    """
    Try to find substructure match, including with atom replacements
    Returns (match_result, modified_substructure_mol, replacement_description, error_details)
    """
    if parent_mol is None or substructure_mol is None:
        return None, None, None, "One or both input molecules are None"
    
    error_log = []
    
    # First try the original substructure matching strategies
    match, error = find_substructure_fast_original(parent_mol, substructure_mol)
    if match is not None:
        return match, substructure_mol, "No replacement needed", "Success - Original substructure"
    
    error_log.append(f"Original substructure failed: {error}")
    
    # Strategy 5: Try with atom replacements on substructure
    try:
        substructure_variants = generate_atom_replacement_variants(substructure_mol)
        
        for variant_mol, replacement_desc in substructure_variants:
            if variant_mol is not None:
                try:
                    # Try all matching strategies with the variant
                    match, variant_error = find_substructure_fast_original(parent_mol, variant_mol)
                    if match is not None:
                        return match, variant_mol, replacement_desc, f"Success - Atom replacement: {replacement_desc}"
                except Exception as e:
                    error_log.append(f"Variant {replacement_desc} failed: {str(e)}")
        
        error_log.append("Strategy 5: All atom replacement variants failed to match")
        
    except Exception as e:
        error_log.append(f"Strategy 5: Atom replacement setup failed - {str(e)}")
    
    return None, None, None, " | ".join(error_log)

def find_substructure_fast(parent_mol, substructure_mol):
    """
    Enhanced substructure matching including atom replacements
    Returns (match_result, modified_substructure, replacement_info, error_details)
    """
    return find_substructure_with_atom_replacements(parent_mol, substructure_mol)

def find_and_mark_substructure_removal_fast(parent_mol, substructure_mol, attachment_point_parent, attachment_point_sub):
    """
    Fast version without MCS - optimized for speed with detailed error tracking and atom replacements
    Returns (remaining_mol, sub_fragment, error_message, replacement_info)
    """
    if parent_mol is None or substructure_mol is None:
        return None, None, "Invalid input molecules - one or both are None", None
    
    # Try to find the substructure match with atom replacements
    match, modified_substructure, replacement_info, match_error = find_substructure_fast(parent_mol, substructure_mol)
    
    if match is None:
        return None, None, f"No substructure match found: {match_error}", None
    
    # Use the modified substructure if atom replacement was successful
    substructure_to_use = modified_substructure if modified_substructure is not None else substructure_mol
    
    try:
        # Create a copy of the parent molecule
        parent_copy = Chem.RWMol(parent_mol)
        
        # Find the bond to break between parent and substructure
        bonds_to_break = []
        
        for bond in parent_mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # Check if bond connects substructure to rest of molecule
            begin_in_match = begin_idx in match
            end_in_match = end_idx in match
            
            if begin_in_match != end_in_match:  # One atom in match, one not
                bonds_to_break.append((begin_idx, end_idx))
        
        if not bonds_to_break:
            return None, None, f"No bonds to break between substructure and parent (match: {match})", replacement_info
        
        # For simplicity, take the first bond to break
        break_bond = bonds_to_break[0]
        begin_idx, end_idx = break_bond
        
        # Determine which atom is in the substructure
        if begin_idx in match:
            sub_atom_idx = begin_idx
            parent_atom_idx = end_idx
        else:
            sub_atom_idx = end_idx
            parent_atom_idx = begin_idx
        
        # Add dummy atom with attachment point to parent molecule
        dummy_atom_parent = Chem.Atom(0)  # Dummy atom (*)
        dummy_atom_parent.SetAtomMapNum(attachment_point_parent)
        dummy_idx_parent = parent_copy.AddAtom(dummy_atom_parent)
        
        # Add bond between dummy atom and the connection point
        parent_copy.AddBond(parent_atom_idx, dummy_idx_parent, Chem.BondType.SINGLE)
        
        # Create substructure fragment with attachment point
        sub_copy = Chem.RWMol(substructure_to_use)
        
        # Add dummy atom to substructure
        dummy_atom_sub = Chem.Atom(0)  # Dummy atom (*)
        dummy_atom_sub.SetAtomMapNum(attachment_point_sub)
        dummy_idx_sub = sub_copy.AddAtom(dummy_atom_sub)
        
        # Find corresponding atom in substructure copy and add bond
        sub_match_dict = {match[i]: i for i in range(len(match))}
        if sub_atom_idx in sub_match_dict:
            sub_copy_atom_idx = sub_match_dict[sub_atom_idx]
            sub_copy.AddBond(sub_copy_atom_idx, dummy_idx_sub, Chem.BondType.SINGLE)
        else:
            return None, None, f"Could not find substructure atom {sub_atom_idx} in match dictionary", replacement_info
        
        # Remove the substructure atoms from parent (in reverse order to maintain indices)
        atoms_to_remove = sorted(match, reverse=True)
        for atom_idx in atoms_to_remove:
            parent_copy.RemoveAtom(atom_idx)
        
        try:
            remaining_mol = parent_copy.GetMol()
            sub_fragment = sub_copy.GetMol()
            
            # Sanitize molecules
            Chem.SanitizeMol(remaining_mol)
            Chem.SanitizeMol(sub_fragment)
            
            # Add replacement info to success message
            success_msg = "Success"
            if replacement_info and replacement_info != "No replacement needed":
                success_msg += f" (with {replacement_info})"
            
            return remaining_mol, sub_fragment, success_msg, replacement_info
        except Exception as e:
            return None, None, f"Sanitization failed: {str(e)}", replacement_info
    
    except Exception as e:
        return None, None, f"Fragment processing failed: {str(e)}", replacement_info

def process_protac_fragmentation_fast(protac_smiles, e3_binder_smiles, ligand_smiles):
    """
    Fast PROTAC fragmentation without expensive MCS with detailed error tracking and atom replacements
    Returns (e3_fragment, warhead_fragment, linker_fragment, detailed_errors)
    """
    detailed_errors = {
        'canonicalization_errors': [],
        'molecule_generation_errors': [],
        'e3_removal_error': None,
        'e3_replacement_info': None,
        'ligand_removal_error': None,
        'ligand_replacement_info': None,
        'hydrogen_addition_errors': []
    }
    
    try:
        # Step 1: Canonicalize all SMILES (fast operation)
        protac_canonical, protac_error = canonicalize_smiles(protac_smiles)
        e3_canonical, e3_error = canonicalize_smiles(e3_binder_smiles)
        ligand_canonical, ligand_error = canonicalize_smiles(ligand_smiles)
        
        if protac_error != "Success":
            detailed_errors['canonicalization_errors'].append(f"PROTAC: {protac_error}")
        if e3_error != "Success":
            detailed_errors['canonicalization_errors'].append(f"E3 Binder: {e3_error}")
        if ligand_error != "Success":
            detailed_errors['canonicalization_errors'].append(f"Ligand: {ligand_error}")
        
        if not all([protac_canonical, e3_canonical, ligand_canonical]):
            return None, None, None, detailed_errors
        
        # Step 2: Generate molecules
        try:
            mol1 = Chem.MolFromSmiles(protac_canonical)
            if mol1 is None:
                detailed_errors['molecule_generation_errors'].append("PROTAC molecule generation failed")
        except Exception as e:
            detailed_errors['molecule_generation_errors'].append(f"PROTAC molecule generation error: {str(e)}")
            mol1 = None
        
        try:
            mol2 = Chem.MolFromSmiles(e3_canonical)
            if mol2 is None:
                detailed_errors['molecule_generation_errors'].append("E3 binder molecule generation failed")
        except Exception as e:
            detailed_errors['molecule_generation_errors'].append(f"E3 binder molecule generation error: {str(e)}")
            mol2 = None
        
        try:
            mol4 = Chem.MolFromSmiles(ligand_canonical)
            if mol4 is None:
                detailed_errors['molecule_generation_errors'].append("Ligand molecule generation failed")
        except Exception as e:
            detailed_errors['molecule_generation_errors'].append(f"Ligand molecule generation error: {str(e)}")
            mol4 = None
        
        if not all([mol1, mol2, mol4]):
            return None, None, None, detailed_errors
        
        # Step 3: Remove E3 binder from PROTAC (fast matching with atom replacements)
        mol3, e3_fragment, e3_error, e3_replacement = find_and_mark_substructure_removal_fast(mol1, mol2, 1, 3)
        detailed_errors['e3_removal_error'] = e3_error
        detailed_errors['e3_replacement_info'] = e3_replacement
        
        if mol3 is None or e3_fragment is None:
            return None, None, None, detailed_errors
        
        # Step 4: Remove ligand from remaining structure (fast matching with atom replacements)
        linker_fragment, warhead_fragment, ligand_error, ligand_replacement = find_and_mark_substructure_removal_fast(mol3, mol4, 2, 4)
        detailed_errors['ligand_removal_error'] = ligand_error
        detailed_errors['ligand_replacement_info'] = ligand_replacement
        
        if linker_fragment is None or warhead_fragment is None:
            return None, None, None, detailed_errors
        
        # Step 5: Add hydrogens at the very end
        e3_fragment, e3_h_error = add_hydrogens_final(e3_fragment)
        warhead_fragment, warhead_h_error = add_hydrogens_final(warhead_fragment)
        linker_fragment, linker_h_error = add_hydrogens_final(linker_fragment)
        
        if e3_h_error != "Success":
            detailed_errors['hydrogen_addition_errors'].append(f"E3 fragment: {e3_h_error}")
        if warhead_h_error != "Success":
            detailed_errors['hydrogen_addition_errors'].append(f"Warhead fragment: {warhead_h_error}")
        if linker_h_error != "Success":
            detailed_errors['hydrogen_addition_errors'].append(f"Linker fragment: {linker_h_error}")
        
        return e3_fragment, warhead_fragment, linker_fragment, detailed_errors
        
    except Exception as e:
        detailed_errors['general_error'] = str(e)
        return None, None, None, detailed_errors

def format_error_details(detailed_errors):
    """Format detailed errors into a readable string"""
    error_parts = []
    
    if detailed_errors.get('canonicalization_errors'):
        error_parts.append(f"Canonicalization: {'; '.join(detailed_errors['canonicalization_errors'])}")
    
    if detailed_errors.get('molecule_generation_errors'):
        error_parts.append(f"Molecule Generation: {'; '.join(detailed_errors['molecule_generation_errors'])}")
    
    if detailed_errors.get('e3_removal_error') and not detailed_errors['e3_removal_error'].startswith("Success"):
        error_parts.append(f"E3 Removal: {detailed_errors['e3_removal_error']}")
    
    if detailed_errors.get('ligand_removal_error') and not detailed_errors['ligand_removal_error'].startswith("Success"):
        error_parts.append(f"Ligand Removal: {detailed_errors['ligand_removal_error']}")
    
    if detailed_errors.get('hydrogen_addition_errors'):
        error_parts.append(f"Hydrogen Addition: {'; '.join(detailed_errors['hydrogen_addition_errors'])}")
    
    if detailed_errors.get('general_error'):
        error_parts.append(f"General Error: {detailed_errors['general_error']}")
    
    return " | ".join(error_parts) if error_parts else "No detailed errors available"

def format_replacement_info(detailed_errors):
    """Format replacement information into a readable string"""
    replacement_parts = []
    
    if (detailed_errors.get('e3_replacement_info') and 
        detailed_errors['e3_replacement_info'] != "No replacement needed" and
        detailed_errors['e3_replacement_info'] is not None):
        replacement_parts.append(f"E3 Binder: {detailed_errors['e3_replacement_info']}")
    
    if (detailed_errors.get('ligand_replacement_info') and 
        detailed_errors['ligand_replacement_info'] != "No replacement needed" and
        detailed_errors['ligand_replacement_info'] is not None):
        replacement_parts.append(f"Ligand: {detailed_errors['ligand_replacement_info']}")
    
    return " | ".join(replacement_parts) if replacement_parts else "No atom replacements"

def main():
    """Main workflow function optimized for speed with detailed error tracking"""
    
    # Load the CSV file
    try:
        df = pd.read_csv('protacdb_20220210.csv')
    except FileNotFoundError:
        print("Error: protacdb_20220210.csv file not found!")
        return
    
    # Check required columns
    required_columns = ['PROTAC SMILES', 'E3 Binder SMILES', 'Ligand SMILES', 'PROTACDB ID', 'Dc50']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Check for additional columns to copy (handle variations in column names)
    additional_columns = []
    column_mapping = {}
    
    # Map actual column names to desired names (handling case variations)
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'ligand pdb' in col_lower or 'pdb ligand' in col_lower:
            additional_columns.append(col)
            column_mapping[col] = 'Ligand_PDB'
        elif 'ligand id' in col_lower:
            additional_columns.append(col)
            column_mapping[col] = 'Ligand_ID'
        elif col_lower == 'pubmed':
            additional_columns.append(col)
            column_mapping[col] = 'Pubmed'
        elif col_lower == 'target':
            additional_columns.append(col)
            column_mapping[col] = 'Target'
    
    print(f"Found additional columns to copy: {additional_columns}")
    if column_mapping:
        print(f"Column mapping: {column_mapping}")
    
    # Initialize results lists
    results = []
    
    print("Processing PROTACs with fast improved fragmentation and detailed error tracking...")
    print("=" * 80)
    
    # Process each row
    for idx, row in df.iterrows():
        protac_id = row['PROTACDB ID']
        dc50 = row['Dc50']
        protac_smiles = row['PROTAC SMILES']
        e3_binder_smiles = row['E3 Binder SMILES']
        ligand_smiles = row['Ligand SMILES']
        
        # Extract additional column values
        additional_data = {}
        for col in additional_columns:
            mapped_name = column_mapping.get(col, col)
            additional_data[mapped_name] = row[col] if col in row else None
        
        # Skip rows with missing SMILES
        if pd.isna(protac_smiles) or pd.isna(e3_binder_smiles) or pd.isna(ligand_smiles):
            result_dict = {
                'PROTACDB ID': protac_id,
                'Dc50': dc50,
                'E3_Binder_Fragment': None,
                'Warhead_Fragment': None,
                'Linker_Fragment': None,
                'Success': False,
                'Error_Details': 'Missing SMILES data (NaN values)',
                'Atom_Replacements': 'N/A'
            }
            # Add additional columns
            result_dict.update(additional_data)
            results.append(result_dict)
            continue
        
        if idx % 100 == 0:
            print(f"Processing PROTAC ID: {protac_id} (row {idx + 1}/{len(df)})")
        
        # Process fragmentation with fast method
        e3_fragment, warhead_fragment, linker_fragment, detailed_errors = process_protac_fragmentation_fast(
            protac_smiles, e3_binder_smiles, ligand_smiles
        )
        
        # Convert fragments to SMILES with attachment points
        e3_smiles = None
        warhead_smiles = None
        linker_smiles = None
        
        try:
            e3_smiles = Chem.MolToSmiles(e3_fragment) if e3_fragment else None
        except Exception as e:
            detailed_errors['e3_smiles_conversion_error'] = str(e)
        
        try:
            warhead_smiles = Chem.MolToSmiles(warhead_fragment) if warhead_fragment else None
        except Exception as e:
            detailed_errors['warhead_smiles_conversion_error'] = str(e)
        
        try:
            linker_smiles = Chem.MolToSmiles(linker_fragment) if linker_fragment else None
        except Exception as e:
            detailed_errors['linker_smiles_conversion_error'] = str(e)
        
        success = all([e3_smiles, warhead_smiles, linker_smiles])
        error_details = format_error_details(detailed_errors) if not success else "Success"
        replacement_info = format_replacement_info(detailed_errors)
        
        # Store results
        result_dict = {
            'PROTACDB ID': protac_id,
            'Dc50': dc50,
            'E3_Binder_Fragment': e3_smiles,
            'Warhead_Fragment': warhead_smiles,
            'Linker_Fragment': linker_smiles,
            'Success': success,
            'Error_Details': error_details,
            'Atom_Replacements': replacement_info
        }
        # Add additional columns
        result_dict.update(additional_data)
        results.append(result_dict)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    total_rows = len(results_df)
    successful_rows = results_df['Success'].sum()
    
    print(f"\nFast Improved Fragmentation Summary with Detailed Error Tracking:")
    print(f"Total PROTACs processed: {total_rows}")
    print(f"Successful fragmentations: {successful_rows}")
    print(f"Success rate: {successful_rows/total_rows*100:.1f}%")
    
    # Save successful fragmentations (include additional columns)
    successful_df = results_df[results_df['Success']].drop(['Success', 'Error_Details'], axis=1)
    output_filename = 'protac_fragments_improved_alt_with_metadata.csv'
    successful_df.to_csv(output_filename, index=False)
    
    print(f"\nSuccessful fragmentations saved to: {output_filename}")
    
    # Save all results (including failures) for debugging with detailed error information
    debug_filename = 'protac_fragments_improved_alt_debug_with_metadata.csv'
    results_df.to_csv(debug_filename, index=False)
    print(f"All results with detailed error tracking saved to: {debug_filename}")
    
    # Print column information
    print(f"\nOutput files contain the following columns:")
    print(f"Core columns: PROTACDB ID, Dc50, E3_Binder_Fragment, Warhead_Fragment, Linker_Fragment, Atom_Replacements")
    if additional_columns:
        mapped_names = [column_mapping.get(col, col) for col in additional_columns]
        print(f"Additional metadata columns: {', '.join(mapped_names)}")
    
    # Print some error statistics and replacement statistics
    failed_df = results_df[~results_df['Success']]
    successful_df_full = results_df[results_df['Success']]
    
    if len(successful_df_full) > 0:
        # Count atom replacements in successful cases
        replacement_counts = {}
        for replacement_info in successful_df_full['Atom_Replacements']:
            if replacement_info != "No atom replacements" and replacement_info != "N/A":
                # Parse the replacement info
                if "E3 Binder:" in replacement_info:
                    replacement_counts['E3 Binder Replacements'] = replacement_counts.get('E3 Binder Replacements', 0) + 1
                if "Ligand:" in replacement_info:
                    replacement_counts['Ligand Replacements'] = replacement_counts.get('Ligand Replacements', 0) + 1
        
        if replacement_counts:
            print(f"\nAtom Replacement Statistics:")
            for replacement_type, count in replacement_counts.items():
                print(f"  {replacement_type}: {count} cases")
            
            # Show some examples of successful replacements
            replacement_examples = successful_df_full[successful_df_full['Atom_Replacements'] != "No atom replacements"]['Atom_Replacements'].head(5)
            if len(replacement_examples) > 0:
                print(f"\nExample successful atom replacements:")
                for i, example in enumerate(replacement_examples, 1):
                    print(f"  {i}. {example}")
    
    if len(failed_df) > 0:
        print(f"\nMost common error types:")
        error_counts = {}
        for error in failed_df['Error_Details']:
            if 'E3 Removal:' in error:
                error_counts['E3 Binder Substructure Not Found'] = error_counts.get('E3 Binder Substructure Not Found', 0) + 1
            elif 'Ligand Removal:' in error:
                error_counts['Ligand Substructure Not Found'] = error_counts.get('Ligand Substructure Not Found', 0) + 1
            elif 'Canonicalization:' in error:
                error_counts['SMILES Canonicalization Error'] = error_counts.get('SMILES Canonicalization Error', 0) + 1
            elif 'Molecule Generation:' in error:
                error_counts['Molecule Generation Error'] = error_counts.get('Molecule Generation Error', 0) + 1
            else:
                error_counts['Other Errors'] = error_counts.get('Other Errors', 0) + 1
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} cases")

if __name__ == "__main__":
    # Suppress RDKit warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()