import re
from typing import List
from ..database.sql_db_tables import CHUNK_INDEXED_COLUMNS

VALID_FIELDS = CHUNK_INDEXED_COLUMNS

def field_match_format(field: str, value: str) -> str:
	"""Format a field match."""
	return f"{field}:{value}"
	# if len(value) >= 2 and value[0] == "\"" and value[-1] == "\"":
	# 	value = value[1:-1]
	# return f"{field} @@@ '{value}'"

def parse_search(text_in: str, 
                 valid_fields: List[str], 
                 catch_all_fields: List[str] = ["text"],
                 return_id_exclusions : bool = False, 
                 return_everything : bool = False) -> str:
	text_in = text_in.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
	
	id_exclusions = []
 
	TWO_SEQUENCE_SLOP = 2
	THREE_SEQUENCE_SLOP = 3
    
	TWO_SEQUENCE_BOOST = 20
	THREE_SEQUENCE_BOOST = 60
    
	assert all([f in valid_fields for f in catch_all_fields]), f"Invalid field(s) {str([f for f in catch_all_fields if not f in valid_fields])} in catch_all_fields. Valid fields are: {valid_fields}"	
 
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

		if not (field in valid_fields or (isinstance(field, str) and field.split(".")[0] in valid_fields)):
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
		
		# TODO: re-enable this.
		# Add slop to phrase query.
		if slop != 1 and term_is_quote_arg and not negative:
			term = f"{term}~{slop}"
		
		# TODO: re-enable this.
		# # Add boost to query.
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
		# TODO: Re-add slop and boost
		args_parsed.append(("\"%s\"~%d^%.2f" % (e, TWO_SEQUENCE_SLOP, TWO_SEQUENCE_BOOST), None))
		# args_parsed.append(("\"%s\"" % e, None))
	
	for e in three_term_sequences:
     	# TODO: Re-add slop and boost
		args_parsed.append(("\"%s\"~%d^%.2f" % (e, THREE_SEQUENCE_SLOP, THREE_SEQUENCE_BOOST), None))
		# args_parsed.append(("\"%s\"" % e, None))
	
	
	p_fields, n_fields = [], []
	
	for (e, field) in args_parsed:
		if field is None:
			p_fields += [field_match_format(catch_all_field, e) for catch_all_field in catch_all_fields]
		else:
			p_fields.append(field_match_format(field, e))
		p_fields = list(set(p_fields))
	for (e, field) in args_parsed_negative:
		if field is None:
			n_fields += [field_match_format(catch_all_field, e) for catch_all_field in catch_all_fields]
		else:
			n_fields.append(field_match_format(field, e))
			if field == "id":
				id_exclusions.append(e.strip("\""))
		n_fields = list(set(n_fields))
	
	negative_field = " NOT ".join(n_fields)
	positive_field = " OR ".join(p_fields)
	# TODO: Fix negative field syntax
	single_condition = f"({positive_field}) NOT {negative_field}" if len(args_parsed_negative) > 0 else f"({positive_field})"
	# single_condition = f"({positive_field})"
	if len(necessary_args) > 0:
		necessary_args = list(set([field_match_format(field, term) for term, field in necessary_args]))
		necessary_args = " AND ".join(necessary_args)
		final_query = f"{necessary_args} AND {single_condition}" \
						if (len(args_parsed) + len(args_parsed_negative)) > 0 \
						else necessary_args
		strong_where_clause = necessary_args + f" NOT {negative_field}" if args_parsed_negative else necessary_args
	else:
		final_query = single_condition
		strong_where_clause = f"NOT {negative_field}" if args_parsed_negative else None
	
 
	if return_id_exclusions:
		return final_query, strong_where_clause, id_exclusions
	elif return_everything:
		return final_query, strong_where_clause, necessary_args, n_fields, p_fields, args_parsed, args_parsed_negative, term_sequences
 
	return final_query, strong_where_clause