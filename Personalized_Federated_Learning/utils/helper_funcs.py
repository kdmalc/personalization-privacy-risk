import ast

def convert_cmd_line_str_lst_to_type_lst(list_as_string, datatype, verbose=False):
    '''
    Safely convert the command line entered list (which Python uses a string representation of) to a list of the correct datatype
    Inputs:
        datatype: int or str, depending on what you're doing
    '''
    #try:
    # Using ast.literal_eval to safely evaluate the string
    if type(list_as_string)!=str:
        #print("CONVERTING TO STRING")
        list_as_string = str(list_as_string)
    #print(f"list_as_string")
    #print(list_as_string)
    temp_lst = ast.literal_eval(list_as_string)
    if datatype==int:
        if not isinstance(temp_lst, list) or not all(isinstance(item, datatype) for item in temp_lst):
            raise ValueError(f"Invalid input. Must be a list of {datatype}.")
    elif datatype==str:
        # Initialize an empty list to store the string elements
        string_list = []

        # Iterate through the characters and add them to the result list as strings
        current_string = ""
        for char in temp_lst:
            #print(f"char: {char}")
            #print(f"current_string: {current_string}")
            #print(f"string_list: {string_list}")
            #print()
            
            if char=='[' or char==']' or char==',' or char==' ':
                #print("MATCH")
                pass
            elif char == "'":
                #print("QUOTATION")
                # When encountering a single quote, it indicates the start or end of a string
                if current_string:
                    string_list.append(current_string)
                    current_string = ""
            else:
                #print("ELSE")
                current_string += char
    else:
        raise ValueError(f"{datatype} is not a supported datatype, please enter <str> or <int> (note: no quotation marks)")
    #except (ValueError, SyntaxError):
    #    print(f"Invalid input. Please provide a valid list of {datatype}.")
    if verbose:
        print(list_as_string)
        print(temp_lst)
    return temp_lst