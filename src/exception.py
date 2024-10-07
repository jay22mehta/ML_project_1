import sys 

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = 'Error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(
    file_name,exc_tb.tb_lineno,str(error))


    return error_message 
    


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Store the detailed error message in an attribute
        self.error_message = self.error_message_detail(error_message, error_detail)

    def error_message_detail(self, error_message, error_detail: sys):
        """
        Create a detailed error message including the file name, line number,
        and the original error message.
        """
        _, _, exc_tb = error_detail.exc_info()  # Get traceback
        file_name = exc_tb.tb_frame.f_code.co_filename  # Get file name
        line_number = exc_tb.tb_lineno  # Get line number

        # Format the error message
        detailed_message = f"Error occurred in script: {file_name}, line: {line_number}, error: {str(error_message)}"
        return detailed_message

    def __str__(self):
        return self.error_message  # Return the detailed error message when printing the exception
