import re
from typing import List
from ..database.sql_db_tables import CHUNK_INDEXED_COLUMNS

VALID_FIELDS = CHUNK_INDEXED_COLUMNS

def parse_search(text_in: str, catch_all_fields: List[str] = ["text"], return_id_exclusions : bool = False):
	text_in = text_in.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
 
	id_exclusions = []
 
	TWO_SEQUENCE_SLOP = 2
	THREE_SEQUENCE_SLOP = 3
    
	TWO_SEQUENCE_BOOST = 3
	THREE_SEQUENCE_BOOST = 5
    
	assert all([f in VALID_FIELDS for f in catch_all_fields]), f"Invalid field in catch_all_fields. Valid fields are: {VALID_FIELDS}"	
 
	call = text_in
	phrase_arguments = {}
	quote_segments = re.finditer(r"\"([^\"]*)\"(\~\d+)?(\^\d+(\.\d+)?)?", call)
	
	for i, segment in enumerate(list(quote_segments)):
		phrase_arguments[f"quote_arg_{i}"] = segment.group(0)
		
	for key, value in phrase_arguments.items():
		call = call.replace(key, f"%{key}")
		call = call.replace(value, key)
	
	call = re.sub(r"(\'|\")", " ", call)
	
	terms = call.split(" ")
	
 
	necessary_args = []
	args_parsed, args_parsed_negative = [], []	
	term_sequences, current_term_sequence = [], []	
	two_term_sequences, three_term_sequences = [], []
	
	# Process terms of the search.
	# Term sequences are used to boost the score of terms that appear together, similar to cover density.
	for term in terms:
		term_is_quote_arg, slop, boost, field = False, 1, 1, None
		
		negative = term.startswith("-")
		term = term.strip("-")

		# print("Looking for field in term:", term)
		field_specified = re.search(r"^([a-zA-Z0-9|\_|\.]+)\:", term)
		if field_specified:
			field = field_specified.group(1)
			term = term[len(field)+1:]

		if not (field in VALID_FIELDS or (isinstance(field, str) and field.split(".")[0] in VALID_FIELDS)):
			field = None

		# Check if the term was moved to a parsed argument, set flag if so.
		if term in phrase_arguments:
			term_is_quote_arg = term
			term : str = phrase_arguments[term]

		boost_specified = re.search(r"\^(\d+(\.\d+)?)$", term)
		if boost_specified:
			boost = float(boost_specified.group(1))
			term = re.sub(r"\^(\d+(\.\d+)?)$", "", term)

		slop_specified = re.search(r"\~(\d+)$", term)
		if slop_specified:
			slop = int(slop_specified.group(1))
			term = re.sub(r"\~(\d+)$", "", term)
		
		# Need to escape them only if they're not already escaped.
		for c in "\\+^`:{}[]()~!*',|&<>?/=":
			# This worked before, try revisiting it
			# term = term.replace(c, "\\%s" % c)
   
			# Wipe the characters. This is a temp fix, because escaping stopped working.
			term = term.replace(c, "")
   
		# If term was a parsed argument, perform preprocessing
		if term_is_quote_arg:
			term = term.strip("\"")
			term = re.sub(r"\"", " ", term) # Probably not necessary
			term = f"\"{term.strip()}\""
		
		if term.strip() in ["", "\"\""]:
			continue
		
		# Add slop to phrase query.
		if slop != 1 and term_is_quote_arg and not negative:
			term = f"{term}~{slop}"
		
		# Add boost to query.
		if boost != 1 and not negative:
			term = f"{term}^{boost}"

		# If isn't negative and isn't a quote sequence, add to current term sequence.
		if (not negative) and (not term_is_quote_arg) and (field is None):
			current_term_sequence.append(term)
		else:
			term_sequences.append(current_term_sequence)
			current_term_sequence = []
		
		# If term is negative, add to negative list.
		if negative:
			args_parsed_negative.append((term, field))
		# If the field is specified, it is a necessary argument.
		elif not field is None:
			necessary_args.append((term, field))
		# If term is positive, add it normally.
		else:
			args_parsed.append((term, field))
	
	term_sequences.append(current_term_sequence)
	
	for term_sequence in term_sequences:
		if len(term_sequence) > 1:
			for i in range(1, len(term_sequence)):
				two_term_sequences.append(f"{term_sequence[i-1]} {term_sequence[i]}")
		if len(term_sequence) > 2:
			for i in range(2, len(term_sequence)):
				three_term_sequences.append(f"{term_sequence[i-2]} {term_sequence[i-1]} {term_sequence[i]}")
	
	for e in two_term_sequences:
		args_parsed.append(("\"%s\"~%d^%.2f" % (e, TWO_SEQUENCE_SLOP, TWO_SEQUENCE_BOOST), None))
	
	for e in three_term_sequences:
		args_parsed.append(("\"%s\"~%d^%.2f" % (e, THREE_SEQUENCE_SLOP, THREE_SEQUENCE_BOOST), None))
	
	
	p_fields, n_fields = [], []
	
	for (e, field) in args_parsed:
		if field is None:
			p_fields += [f"{catch_all_field}:{e}" for catch_all_field in catch_all_fields]
		else:
			p_fields.append(f"{field}:{e}")
	
	for (e, field) in args_parsed_negative:
		if field is None:
			n_fields += [f"{catch_all_field}:{e}" for catch_all_field in catch_all_fields]
		else:
			n_fields.append(f"{field}:{e}")
			if field == "id":
				id_exclusions.append(e.strip("\""))
	
	negative_field = " NOT ".join(n_fields)
	positive_field = " OR ".join(p_fields)
	single_condition = f"({positive_field}) NOT {negative_field}" if len(args_parsed_negative) > 0 else f"({positive_field})"
	if len(necessary_args) > 0:
		necessary_args = [f"{field}:{term}" for term, field in necessary_args]
		necessary_args = " AND ".join(necessary_args)
		final_query = f"{necessary_args} AND {single_condition}" \
      					if (len(args_parsed) + len(args_parsed_negative)) > 0 \
           				else necessary_args
	else:
		final_query = single_condition
	
	if return_id_exclusions:
		return final_query, id_exclusions
 
	return final_query