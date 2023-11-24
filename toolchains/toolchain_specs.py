example_toolchain = {
	"name": "Breast Cancer Staging",
	"chat_window_settings": {
		"websearch": False,
        "file_upload": {
            "value": True,
            "multiple": True
        },
        "display_mode": "single_table",
        "store_session": True,
        "download_results": True
	},
    "pipeline_starting_input": {
        "arguments": ["user_files"]
    },
    "pipeline": [
        {
            "layer_id": "user_inputs",
            "function": None,
            "arguments": [
                {
                    "input": "user_files",
                    "type": "file",
                    "iterable": True,
                    "optional": False
                }
            ],
            "execution": "synchronous",
            "stream_output": False,
            "iterate": True,
            "output_type": "json",
            "feed_to": [
                {
                    "destination": "staging",
                    "split_outputs": True,
                    "target_argument": "file_input"
                }
            ]
        },
        {
            "layer_id": "staging",
            "function": "stage_breast_cancer_report",
            "arguments": [
                {
                    "argument_name": "file_input",
                    "type": "file",
                    "iterable_sources": False,
                    "optional": False
                }
            ],
            "execution": "synchronous",
            "stream_output": False,
            "iterate": True,
            "output_type": "json",
            "feed_to": [
                {
                    "destination": "table_formatting",
                    "split_outputs": False,
                    "target_argument": "table_row"
                }
            ]
        },
        {
            "layer_id": "table_formatting",
            "function": "format_jsons_to_table",
            "arguments": [
                {
                    "argument_name": "table_row",
                    "type": "json",
                    "iterable_sources": True,
                    "optional": False
                }
            ],
            "stream_input": True,
            "execution": "asynchronous",
            "iterate": False,
            "output_type": "string",
            "feed_to": [
                {
                    "destination": "user",
                    "argument_class": "display_table",
                    "stream": True,
                    "stream_type": "total_revision"
                },
                {
                    "destination": "download_formats",
                    "target_argument": "markdown_table",
                    "stream": False
                }
            ]
        },
        {
            "layer_id": "download_formats",
            "function": "convert_markdown_table_to_data_formats",
            "arguments": [
                {
                    "argument_name": "markdown_table",
                    "type": "string",
                    "iterable_sources": False,
                    "optional": False
                }
            ],
            "stream_input": False,
            "execution": "synchronous",
            "iterate": False,
            "output_type": "list",
            "feed_to": [
                {
                    "destination": "user",
                    "argument_class": "download_links",
                    "stream": False
                }
            ]
        }
    ]
}