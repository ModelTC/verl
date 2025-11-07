def merge_messages(messages: list[dict]):
    merged_messages = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            merged_content = ''
            for item in content:
                if item['type'] == 'text':
                    merged_content += item['text']
            content = merged_content
        merged_messages.append({
            "role": message['role'],
            "content": content
        })
    return merged_messages