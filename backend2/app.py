from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS 
import re
import ast
import docx
import os
import requests
import pandas as pd
import pubchempy as pcp
from werkzeug.utils import secure_filename
import json
import time
from fuzzywuzzy import fuzz, process

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'docx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(app)


def extract_ingredients_from_text(text):
    # Extract content between square brackets
    pattern = r'\[(.*?)\]$'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        pattern = r'\[(.*)'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return []
    
    content = match.group(1)
    
    # Remove unwanted content
    unwanted_patterns = [
        r'\(\*\) Additional labeling.*',
        r'\(\*\) Exigence d.*',
        r'1223/2009.*'
    ]
    
    for pattern in unwanted_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Clean the content
    content = content.replace('\xa0', ' ').replace('\"', '').strip()
    
    # Parse the string as a Python list using ast.literal_eval
    try:
        items = ast.literal_eval(f"[{content}]")
    except:
        # Fallback: split on commas but handle quoted strings properly
        items = []
        in_quotes = False
        current_item = ""
        
        for char in content:
            if char == "'" and not in_quotes:
                in_quotes = True
                current_item += char
            elif char == "'" and in_quotes:
                in_quotes = False
                current_item += char
                items.append(current_item.strip())
                current_item = ""
            elif in_quotes:
                current_item += char
            elif char == ',' and not in_quotes:
                if current_item.strip():
                    items.append(current_item.strip())
                current_item = ""
            else:
                current_item += char
        
        if current_item.strip():
            items.append(current_item.strip())
    
    ingredients = []
    start_markers = ["% WW / %P/P", "% WW/%P/P", "% W/W / %P/P", "INGREDIENTS", "INGREDIENTS/INGREDIENTES"]
    
    # Remove quotes from items for easier processing
    cleaned_items = [item.strip("'") for item in items]
    
    # Find all start marker positions
    start_positions = []
    for i, item in enumerate(cleaned_items):
        if any(marker == item for marker in start_markers):
            start_positions.append(i)
    
    if start_positions:
        # Use the last start marker (most specific)
        start_index = max(start_positions)
        
        # Collect items after the start marker, skipping percentage values
        i = start_index + 1
        while i < len(cleaned_items):
            item = cleaned_items[i]
            
            # Skip percentage values (numbers with comma as decimal separator)
            if re.match(r'^\d+,\d+$', item) or re.match(r'^\d+$', item) or item == 'QSP 100':
                i += 1
                continue
                
            # If we have an ingredient name, add it
            if item.strip() and not any(marker == item for marker in start_markers):
                ingredients.append(item)
            
            i += 1
    else:
        for item in cleaned_items:
            if (not re.match(r'^\d+,\d+$', item) and 
                not re.match(r'^\d+$', item) and 
                item != 'QSP 100' and
                not any(marker == item for marker in start_markers)):
                ingredients.append(item)
    
    return ingredients

def get_pubchem_info(ingredient_name):
    try:
        compounds = pcp.get_compounds(ingredient_name, 'name')
        if compounds:
            compound = compounds[0]
            return {
                'cid': compound.cid,
                'name': compound.iupac_name or compound.synonyms[0] if compound.synonyms else ingredient_name,
                'molecular_formula': compound.molecular_formula,
                'molecular_weight': compound.molecular_weight,
                'is_valid': True
            }
        else:
            return {
                'name': ingredient_name,
                'is_valid': False,
                'error': 'Not found in PubChem'
            }
    except Exception as e:
        return {
            'name': ingredient_name,
            'is_valid': False,
            'error': str(e)
        }

def read_docx_file(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error: {str(e)}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def clean_ingredient_name(name):
    """Clean ingredient name for better matching - remove all non-alphanumeric characters"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', name)
    return cleaned.upper()

def find_similarity_match(doc1_ing, doc2_ingredients, threshold=80):
    """Find similarity match using fuzzy string matching"""
    # Try exact match first
    if doc1_ing in doc2_ingredients:
        return doc1_ing, 100, 'EXACT'
    
    # Try partial ratio for substring-like matching
    best_match, best_score = None, 0
    for doc2_ing in doc2_ingredients:
        score = fuzz.partial_ratio(doc1_ing.upper(), doc2_ing.upper())
        if score > best_score:
            best_score = score
            best_match = doc2_ing
    
    if best_score >= threshold:
        return best_match, best_score, 'SIMILARITY'
    
    return None, 0, 'NOT_FOUND'

def find_individual_ingredient_match(doc1_ing, doc2_ingredients):
    """Check if doc1 ingredient exists as individual word in any doc2 ingredient"""
    doc1_clean = clean_ingredient_name(doc1_ing)
    
    for doc2_ing in doc2_ingredients:
        # Split doc2 ingredient by common separators to check individual words
        doc2_parts = re.split(r'[\s\/\.\,\(\)]', doc2_ing)
        doc2_parts = [part.strip() for part in doc2_parts if part.strip()]
        
        # Check each individual part
        for part in doc2_parts:
            part_clean = clean_ingredient_name(part)
            if doc1_clean == part_clean:
                return doc2_ing, 'CONTAINMENT'
        
        # Also check if doc1 ingredient is contained in the whole doc2 ingredient
        doc2_clean = clean_ingredient_name(doc2_ing)
        if doc1_clean in doc2_clean:
            return doc2_ing, 'CONTAINMENT'
    
    return None, 'NOT_FOUND'

def compare_documents_logic(file1_path, file2_path):
    doc1_text = read_docx_file(file1_path)
    doc2_text = read_docx_file(file2_path)

    if doc1_text.startswith('Error') or doc2_text.startswith('Error'):
        return {'error': 'Error reading DOCX files'}
    
    doc1_ingredients = extract_ingredients_from_text(doc1_text)
    doc2_ingredients = extract_ingredients_from_text(doc2_text)

    # Get all unique ingredients for comparison table
    all_ingredients = sorted(set(doc1_ingredients + doc2_ingredients))
    comparison_data = []
    valid_ingredients = []

    # First pass: Check exact matches
    exact_matches = set(doc1_ingredients) & set(doc2_ingredients)
    
    # Second pass: Check individual word matches in compound ingredients
    containment_match = {}
    containment_match_types = {}
    for doc1_ing in doc1_ingredients:
        if doc1_ing not in exact_matches:
            match, match_type = find_individual_ingredient_match(doc1_ing, doc2_ingredients)
            if match:
                containment_match[doc1_ing] = match
                containment_match_types[doc1_ing] = match_type

    # Third pass: Check similarity matches for remaining ingredients
    similarity_matches = {}
    similarity_scores = {}
    for doc1_ing in doc1_ingredients:
        if doc1_ing not in exact_matches and doc1_ing not in containment_match:
            best_match, best_score, match_type = find_similarity_match(doc1_ing, doc2_ingredients, threshold=80)
            if best_match:
                similarity_matches[doc1_ing] = best_match
                similarity_scores[doc1_ing] = best_score

    # Build comparison data
    for ingredient in all_ingredients:
        # Check presence in doc1 (formula)
        in_doc1_exact = ingredient in doc1_ingredients
        
        # Check presence in doc2 (snapshot)
        in_doc2_exact = ingredient in doc2_ingredients
        in_doc2_individual = ingredient in containment_match
        in_doc2_similarity = ingredient in similarity_matches
        
        # Final determination - in doc2 if any type of match
        in_doc2_final = in_doc2_exact or in_doc2_individual or in_doc2_similarity
        
        # Determine match type and details
        if in_doc2_exact:
            match_type = 'EXACT'
            matched_in = ingredient
            similarity_score = 100
        elif in_doc2_individual:
            match_type = containment_match_types[ingredient]
            matched_in = containment_match[ingredient]
            similarity_score = 100
        elif in_doc2_similarity:
            match_type = 'SIMILARITY'
            matched_in = similarity_matches[ingredient]
            similarity_score = similarity_scores[ingredient]
        else:
            match_type = 'NOT_FOUND'
            matched_in = ''
            similarity_score = 0
        
        ingredient_data = {
            'original_name': ingredient,
            'formula': 'YES' if in_doc1_exact else 'NO',
            'snapshot': 'YES' if in_doc2_final else 'NO',
            'match_type': match_type,
            'matched_in': matched_in,
            'similarity_score': similarity_score
        }
        
        comparison_data.append(ingredient_data)
        
        if ingredient_data['formula'] == "YES":
            valid_ingredients.append(ingredient_data)

    total_compared = len(doc1_ingredients)
     # Count individual match types
    containment_count = sum(1 for ing in containment_match if containment_match_types[ing] == 'CONTAINMENT')
    matched = sum(1 for item in comparison_data if item['formula'] == item['snapshot'])
    mismatched = total_compared - matched
    mismatch_percentage = (mismatched / total_compared) * 100 if total_compared > 0 else 0
    # Calculate detailed statistics
    exact_match_count = len(exact_matches)
    individual_match_count = len(containment_match)
    similarity_match_count = len(similarity_matches)
    match_percentage = ((exact_match_count + containment_count + similarity_match_count) / total_compared) * 100 if total_compared > 0 else 0
    status = "PASSED" if match_percentage == 100 else "FAILED"


    response = {
        'valid_ingredients_table': valid_ingredients,
        'summary': {
            'total_compared': total_compared,
            'matched': matched,
            'match_percentage': round(match_percentage),
            'mismatched': mismatched,
            'status': status,
            'mismatch_percentage': round(mismatch_percentage, 2),
            'exact_matches': exact_match_count,
            'containment_matches': containment_count,
            'similarity_matches': similarity_match_count,
        },
        'counts': {
            'formula_count': len(doc1_ingredients),
            'snapshot_count': len(doc2_ingredients),
            'common_exact_count': exact_match_count,
            'common_containment_count': individual_match_count,
            'common_similarity_count': similarity_match_count,
            'total_common_count': exact_match_count + individual_match_count + similarity_match_count
        }
    }
    return response

@app.route('/', methods=['GET'])
def index():
   return jsonify({"message": "Backend is running Fine"})

@app.route('/compare', methods=['POST'])
def compare_documents():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Please upload both files'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Please select both files'}), 400

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(filepath1)
        file2.save(filepath2)

        results = compare_documents_logic(filepath1, filepath2)

        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except:
            pass

        if 'error' in results:
            return jsonify({'error': results['error']}), 400
        return jsonify(results)
    else:
        return jsonify({'error': 'Only DOCX files are allowed'}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
