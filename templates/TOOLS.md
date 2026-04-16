# Tool Usage Notes

Tool signatures are provided automatically via function calling.

## Handling Tool Failures

When a tool returns an error or no results:

1. **Verify the failure**: Check if the error is real or the tool actually returned usable data
2. **Do NOT give up easily**: Most queries can be answered with persistent effort
3. **Report partial results**: If you get some data, share it and note limitations

### Data Modality Selection

- Images: use image_* tools
- Point clouds: use pointcloud_* tools  
- Time series: use timeseries_* tools
- Tabular data: use data_cleaner, data_transformer tools

### Byte-Level Processing

For tasks involving raw bytes (e.g., extracting last N bytes, sorting bytes, merging binary data):
- Use **read_file** to read file content (supports binary mode)
- Use **exec** with Python for byte manipulation: `python3 -c "..."`
- Use **write_file** to save results

Example workflow for "extract last 100 bytes from all files, sort and merge":
1. Use **list_dir** to find all files in the target directory
2. Use **read_file** or **exec** to extract last 100 bytes from each file
3. Use **exec** with Python to sort and merge bytes: `python3 -c "import sys; bytes_list = [...]; sorted_bytes = sorted(bytes_list); open('output.bin', 'wb').write(bytes(sorted_bytes))"`
4. Use **write_file** or confirm with user

## spawn

- Use the built-in **spawn** tool to launch background sub‑agents. For tasks such as data processing, data retrieval, and batch file operations, always use spawn to start the TaskPlanner.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters

## cron — Scheduled Tasks

Use the built-in `cron` tool for periodic data processing tasks.
