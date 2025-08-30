class InvalidArgumentError(ValueError):
    """Custom exception raised when a function argument is not in valid values."""
    
    def __init__(self, argument_name, provided_value, valid_values):
        self.argument_name = argument_name
        self.provided_value = provided_value
        self.valid_values = valid_values
        
        # Create a detailed error message
        valid_str = ", ".join(map(str, valid_values))
        message = (f"Invalid value for argument '{argument_name}': "
                  f"'{provided_value}'. Valid values are: [{valid_str}]")
        
        super().__init__(message)


class NonOverlappingData(Exception):
    """Custom exception raised when the provided dataframes have no overlapping timestamps."""
    
    def __init__(self):
        message = "There is no time overlap in the provided dataframes."
        super().__init__(message)


class NotAvailableData(Exception):
    """Custom exception raised when the provided dataframes have no data related to the index.
        data_frame_scale : [higher, lower]
    """
    
    def __init__(self, index:int, data_frame_scale:str):
        message = f"There is no data point related to index={index} in the {data_frame_scale} dataframe."
        super().__init__(message)
