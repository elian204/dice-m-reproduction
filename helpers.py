# Standard library imports
import os
from typing import Dict, Tuple, Optional, Any, Union
import pathlib
from aaai_experiments import *

# Third-party imports
import numpy as np
import pandas as pd
from lxml import etree

# Process mining specific imports
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.petri_net.obj import Marking


def convert_log_to_xes(df: pd.DataFrame, output_file: str) -> None:  
    """
    Converts a Pandas DataFrame representing a process log into an XES file without namespaces.
    
    Parameters:
        df (pd.DataFrame): The input log with columns 'concept:name' (activity names) 
                          and 'case:concept:name' (case IDs).
        output_file (str): The name of the output XES file.
    
    Returns:
        None: Saves the converted log as an XES file without namespaces.
        
    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    required_columns = {'concept:name', 'case:concept:name'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Convert the DataFrame to a pm4py Event Log
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    event_log = log_converter.apply(df)
    
    # Export the Event Log to a temporary XES file
    temp_file = "temp_with_namespace.xes"
    try:
        xes_exporter.apply(event_log, temp_file)
        
        # Parse and clean the XML
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(temp_file, parser)
        root = tree.getroot()
        
        # Create new root element without namespace
        new_root = etree.Element("log")
        
        # Add XES version attribute
        new_root.attrib["xes.version"] = "1.0"
        new_root.attrib["xes.features"] = "nested-attributes"
        
        # Add classifier
        classifier = etree.SubElement(new_root, "classifier")
        classifier.attrib["name"] = "Activity classifier"
        classifier.attrib["keys"] = "concept:name"
        
        # Add standard extensions
        extensions = [
            ("Concept", "http://www.xes-standard.org/concept.xesext"),
            ("Time", "http://www.xes-standard.org/time.xesext"),
            ("Organizational", "http://www.xes-standard.org/org.xesext")
        ]
        
        for ext_name, ext_value in extensions:
            extension = etree.SubElement(new_root, "extension")
            extension.attrib["name"] = ext_name
            extension.attrib["prefix"] = ext_name.lower()
            extension.attrib["uri"] = ext_value
        
        def clean_element(elem: etree._Element) -> Dict[str, Any]:
            """Helper function to clean element attributes and get clean data"""
            clean_data = {}
            clean_data["tag"] = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            clean_data["attrib"] = {
                k.split("}")[-1] if "}" in k else k: v
                for k, v in elem.attrib.items()
            }
            return clean_data
        
        # Process traces and events
        for trace_elem in root.findall(".//{*}trace"):
            new_trace = etree.SubElement(new_root, "trace")
            
            # Add trace attributes
            trace_data = clean_element(trace_elem)
            for key, value in trace_data["attrib"].items():
                if key != "xmlns":
                    new_trace.attrib[key] = value
                    
            # Process events within trace
            for event_elem in trace_elem.findall(".//{*}event"):
                new_event = etree.SubElement(new_trace, "event")
                event_data = clean_element(event_elem)
                
                # Add event attributes
                for key, value in event_data["attrib"].items():
                    if key != "xmlns":
                        new_event.attrib[key] = value
                        
                # Process event children (string, date, etc.)
                for child in event_elem:
                    child_data = clean_element(child)
                    new_child = etree.SubElement(new_event, child_data["tag"])
                    for key, value in child_data["attrib"].items():
                        if key != "xmlns":
                            new_child.attrib[key] = value
                    if child.text and child.text.strip():
                        new_child.text = child.text.strip()
        
        # Create new tree with clean root
        new_tree = etree.ElementTree(new_root)
        
        # Write the cleaned XES file
        new_tree.write(
            output_file,
            pretty_print=True,
            encoding="UTF-8",
            xml_declaration=True
        )
        print(f"Log successfully converted and saved to {output_file} without namespaces.")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def sample_traces(df: pd.DataFrame, num_traces: int, random_state: int = None) -> pd.DataFrame:
    """
    Randomly samples a specified number of unique traces from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the event log.
    - num_traces (int): The number of unique traces to sample.
    - random_state (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
    - pd.DataFrame: A DataFrame containing the sampled traces.
    """
    # Ensure the DataFrame has the necessary columns
    if not {'concept:name', 'case:concept:name'}.issubset(df.columns):
        raise ValueError("The DataFrame must contain 'concept:name' and 'case:concept:name' columns.")

    # Extract unique trace identifiers
    unique_traces = df['case:concept:name'].unique()

    # Check if the requested number of traces exceeds the available unique traces
    if num_traces > len(unique_traces):
        raise ValueError(f"Requested {num_traces} traces, but only {len(unique_traces)} unique traces are available.")

    # Sample the specified number of unique traces
    sampled_trace_ids = pd.Series(unique_traces).sample(n=num_traces, random_state=random_state).tolist()

    # Filter the DataFrame to include only the sampled traces
    sampled_df = df[df['case:concept:name'].isin(sampled_trace_ids)]

    return sampled_df


def prepare_event_log(
    df_name: str,
    n_train_traces: int,
    n_test_traces: int,
    min_len: int = 1,
    max_len: Optional[int] = None,
    n_traces: Optional[int] = None,
    random_seed: int = 42,
    data_path: str = "./data",
    subfolder: str = "",
    max_samples_per_activity: Optional[int] = None,
    print_dataset_stats: bool = False,
    stats: Optional[Dict[str, Any]] = None,
    allow_variant_intersection: bool = True,
    unique_train_variants: bool = False,
    unique_test_variants: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Prepare event log data by loading, preprocessing, and splitting into train and test sets.
    Args:
        df_name: Name of the event log file
        n_train_traces: Number of traces for training set
        n_test_traces: Number of traces for test set
        min_len: Minimum length of traces to include
        max_len: Maximum length of traces to include
        n_traces: Total number of traces to consider
        random_seed: Random seed for reproducibility
        data_path: Path to the data directory
        subfolder: Subfolder within data directory
        max_samples_per_activity: Maximum samples per activity
        print_dataset_stats: Whether to print dataset statistics
        stats: Dictionary to store statistics if print_dataset_stats is True
        allow_variant_intersection: Whether to allow variant intersection between train and test
        unique_train_variants: Whether to ensure unique variants in training set
        unique_test_variants: Whether to ensure unique variants in test set
    Returns:
        Tuple containing:
        - Training DataFrame
        - Test DataFrame
        - Mapping dictionary from preprocessing
    """
    # Load and preprocess the log
    df, map_dict = load_and_preprocess_log(
        df_name,
        min_len=min_len,
        max_len=max_len,
        n_traces=n_traces,
        random_seed=random_seed,
        path=data_path,
        subfolder=subfolder,
        max_samples_per_activity=max_samples_per_activity,
        stats=stats if print_dataset_stats else None
    )
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Split the log into train and test sets
    result = train_test_log_split_simplified(
        df, 
        n_train_traces=n_train_traces, 
        n_test_traces=n_test_traces,
        random_seed=random_seed,
        allow_variant_intersection=allow_variant_intersection,
        unique_train_variants=unique_train_variants,
        unique_test_variants=unique_test_variants
    )
    
    train_df, test_df = result['train_df'], result['test_df']
    
    # Calculate and print trace statistics
    def get_trace_stats(df: pd.DataFrame) -> Tuple[float, int]:
        """Calculate average trace length and number of unique traces.
        
        Args:
            df: DataFrame containing the event log
            
        Returns:
            Tuple containing:
            - Average trace length
            - Number of unique traces
        """
        trace_lengths = df.groupby('case:concept:name').size()
        avg_length = trace_lengths.mean()
        unique_traces = len(df['case:concept:name'].unique())
        return avg_length, unique_traces
    
    train_avg_len, train_unique = get_trace_stats(train_df)
    test_avg_len, test_unique = get_trace_stats(test_df)
    
    print(f"Train set statistics:")
    print(f"- Average trace length: {train_avg_len:.2f}")
    print(f"- Number of unique traces: {train_unique}")
    print(f"\nTest set statistics:")
    print(f"- Average trace length: {test_avg_len:.2f}")
    print(f"- Number of unique traces: {test_unique}")
    
    return train_df, test_df, map_dict


def xes_to_dataframe(xes_path: str) -> pd.DataFrame:
    """
    Converts a log in XES format into a Pandas DataFrame.
    
    Args:
        xes_path (str): Path to the .xes file.
    
    Returns:
        pd.DataFrame: DataFrame containing the event log with columns such as
                      'case:concept:name', 'concept:name', 'time:timestamp', etc.
    """
    # Import the log using pm4py
    log = xes_importer.apply(xes_path)
    
    # Convert the event log into a list of dictionaries
    events = []
    for trace in log:
        case_id = trace.attributes['concept:name']
        for event in trace:
            event_data = {key: event.get(key) for key in event.keys()}
            event_data['case:concept:name'] = case_id
            events.append(event_data)
    
    # Convert the list of events to a DataFrame
    df = pd.DataFrame(events)
    return df


def sample_traces(
    df: pd.DataFrame,
    n_traces: Union[int, float],
    random_state: int = None
) -> pd.DataFrame:
    """
    Sample a specified number of traces from an event log dataframe while maintaining
    the sequential order of activities within each trace.

    Args:
        df (pd.DataFrame): Input event log dataframe with 'case:concept:name' column
        n_traces (Union[int, float]): Number of traces to sample. If int, samples exact number
            of traces. If float between 0 and 1, samples that fraction of traces.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A new dataframe containing only the sampled traces with their
            activities in original order.

    Raises:
        ValueError: If n_traces is larger than the number of unique traces in the log
            or if n_traces is not a positive number.
    """
    # Get unique case IDs
    unique_cases = df["case:concept:name"].unique()
    n_unique_cases = len(unique_cases)

    # Validate inputs
    if isinstance(n_traces, float):
        if not 0 < n_traces <= 1:
            raise ValueError("If n_traces is a float, it must be between 0 and 1")
        n_traces = int(n_unique_cases * n_traces)
    elif not isinstance(n_traces, int) or n_traces <= 0:
        raise ValueError("n_traces must be a positive integer or float between 0 and 1")
    if n_traces > n_unique_cases:
        raise ValueError(f"Cannot sample {n_traces} traces; only {n_unique_cases} available")

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Sample case IDs
    sampled_cases = np.random.choice(unique_cases, size=n_traces, replace=False)

    # Filter dataframe to only include sampled cases and preserve original order
    return df[df["case:concept:name"].isin(sampled_cases)].copy()


def save_log_as_csv(
    df: pd.DataFrame,
    output_path: Union[str, pathlib.Path],
    encoding: str = 'utf-8',
    separator: str = ',',
    include_index: bool = False
) -> None:
    """
    Convert case:concept:name and concept:name columns to string type and save DataFrame as CSV.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be processed and saved
        output_path (Union[str, pathlib.Path]): Path where the CSV will be saved
        encoding (str, optional): File encoding. Defaults to 'utf-8'
        separator (str, optional): CSV separator character. Defaults to ','
        include_index (bool, optional): Whether to include index in CSV. Defaults to False
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Create a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    
    # Convert specific columns to string
    columns_to_convert = ['case:concept:name', 'concept:name']
    for col in columns_to_convert:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype('string')
    
    processed_df.to_csv(
        output_path,
        index=include_index,
        encoding=encoding,
        sep=separator,
        mode='w'
    )


def export_petri_net(train_df, output_pnml_file):
    """
    Discovers a Petri net from the given DataFrame, creates proper initial and final markings,
    and exports the model as a PNML file.
    
    Parameters:
        train_df (pd.DataFrame): Input DataFrame for process discovery.
        output_pnml_file (str): The file path where the PNML file will be saved.
    """
    # Prepare the DataFrame (ensure you have defined prepare_df_cols_for_discovery)
    train_df = prepare_df_cols_for_discovery(train_df)
    
    # Discover the Petri net using the inductive miner
    net, original_initial_marking, original_final_marking = pm4py.discover_petri_net_inductive(train_df)
    
    # Create new proper markings
    initial_marking = Marking()
    final_marking = Marking()
    
    # Find the source and sink places and properly mark them
    for place in net.places:
        if str(place) == 'source':  # or use place.name if available
            initial_marking[place] = 1
        elif str(place) == 'sink':  # or use place.name if available
            final_marking[place] = 1
    
    # Export the Petri net model as a PNML file
    pnml_exporter.apply(net, initial_marking, output_pnml_file, final_marking=final_marking)
    print(f"Process model exported to {output_pnml_file}")


# Originial
def export_event_log(test_df, output_xes_file, parameters=None):
    """
    Converts a DataFrame into an event log using provided parameters and exports it as an XES file.
    
    Parameters:
        test_df (pd.DataFrame): DataFrame containing event log data.
        output_xes_file (str): The file path where the XES file will be saved.
        parameters (dict, optional): Mapping for column names, e.g., 
                                     {"case_id_key": "case:concept:name", "activity_key": "concept:name"}.
                                     If None, defaults will be used.
    """
    # Set default parameters if none are provided
    if parameters is None:
        parameters = {
            "case_id_key": "case:concept:name",  # adjust these keys based on your DataFrame
            "activity_key": "concept:name"
        }
    
    # Convert the DataFrame to an event log
    event_log = log_converter.apply(test_df, parameters=parameters)
    
    # Export the event log as an XES file
    xes_exporter.apply(event_log, output_xes_file)
    print(f"Event log exported to {output_xes_file}")


def export_existing_petri_net(process_model, output_pnml_file: str):
    """
    Exports an existing Petri net process model as a PNML file.
    
    The input process model must have the following attributes:
        - pm4py_net: The underlying Petri net.
        - pm4py_initial_marking: (Optional) The original initial marking.
        - pm4py_final_marking: (Optional) The original final marking.
    
    Instead of using the existing markings, this function creates new proper markings
    by iterating over net.places to identify the source and sink places.
    
    Parameters:
        process_model: An object representing the process model with the required pm4py attributes.
        output_pnml_file (str): The file path where the PNML file will be saved.
        
    Raises:
        ValueError: If the process model does not contain the required 'pm4py_net' attribute.
        AttributeError: If the process model does not have a 'places' attribute to derive markings from.
    """
    # Validate that the process_model contains the required field for the net.
    if not hasattr(process_model, "pm4py_net"):
        raise ValueError("The process model does not contain the required 'pm4py_net' attribute.")
    
    net = process_model.pm4py_net

    # Create new proper markings
    initial_marking = Marking()
    final_marking = Marking()
    
    # Ensure that the net has places to create markings.
    if not hasattr(net, "places"):
        raise AttributeError("The Petri net does not have a 'places' attribute to derive markings from.")
    
    # Find the source and sink places and properly mark them.
    for place in net.places:
        if str(place) == 'source':  # or use place.name if available
            initial_marking[place] = 1
        elif str(place) == 'sink':  # or use place.name if available
            final_marking[place] = 1

    # Export the Petri net as a PNML file
    pnml_exporter.apply(net, initial_marking, output_pnml_file, final_marking=final_marking)
    print(f"Process model exported to {output_pnml_file}")


def load_process_model_with_learned_map(
    model_path: str,
    dataset_path: str,
    min_len: int = None,
    max_len: int = None,
    n_traces: int = None,
    random_seed: int = 304,
    subfolder: str = None,
    max_samples_per_activity: int = None,
    stats: dict = None,
    return_markings: bool = False
):
    """
    Reads a process model from a file after learning the activity mapping dictionary
    from a dataset provided via a file path. The dataset is preprocessed using a helper
    function (e.g., load_and_preprocess_log), which returns both the preprocessed dataset
    and the learned mapping dictionary.

    Args:
        model_path (str): The file path to the process model.
        dataset_path (str): The file path to the dataset for learning the mapping dictionary.
        min_len (int, optional): Minimum trace length to include during preprocessing.
        max_len (int, optional): Maximum trace length to include during preprocessing.
        n_traces (int, optional): Total number of traces to use during preprocessing.
        random_seed (int): Seed for reproducibility (default: 304).
        subfolder (str, optional): Subfolder within the data path, if applicable.
        max_samples_per_activity (int, optional): Maximum samples per activity.
        stats (dict, optional): Dictionary to store statistics collected during preprocessing.
        return_markings (bool): Whether to return the process model with markings. Defaults to False.

    Returns:
        The process model object generated by `generate_model_from_file`.

    Raises:
        ValueError: If the dataset_path is not provided or if the mapping dictionary cannot be learned.
    """
    if not dataset_path:
        raise ValueError("A dataset_path must be provided to learn the activity mapping dictionary.")

    # Preprocess the dataset to learn the mapping dictionary.
    # Assume load_and_preprocess_log returns a tuple (df, map_dict).
    _, map_dict = load_and_preprocess_log(
        dataset_path,
        min_len=min_len,
        max_len=max_len,
        n_traces=n_traces,
        random_seed=random_seed,
        subfolder=subfolder,
        max_samples_per_activity=max_samples_per_activity,
        stats=stats
    )
    
    if not map_dict:
        raise ValueError("Failed to learn the activity mapping dictionary from the dataset.")

    # Generate the process model using the learned mapping dictionary.
    model = generate_model_from_file(
        model_path,
        activity_mapping_dict=map_dict,
        return_markings=return_markings
    )
    
    return model