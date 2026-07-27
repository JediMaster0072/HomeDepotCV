import re

from translators.atomic.entity_translators import SKUEntityTranslator


# Pipeline order: Optional legacy fallback
# Description: Extracts SKU-like and vendor-SKU-like strings from legacy Google OCR text annotations.
def find_sku_entities(bounding_text, store_number, file_name=None):
    sku_entity_translator = SKUEntityTranslator("skus")
    vendor_sku_entity_translator = SKUEntityTranslator("vendor_skus")
    recovered_skus = set()  # Track SKUs recovered from OCR edge cases

    # Handle both string input (test) and bounding_text object (production)
    if isinstance(bounding_text, str):
        # Test case: string input
        combined_text = bounding_text
    else:
        # Production case: extract from text_annotations to preserve duplicates
        # Skip text_annotations[0] which contains the ENTIRE image text
        # Only use text_annotations[1:] which are individual words/phrases
        # This prevents double-counting SKUs (once from full text, once from individual words)
        texts = bounding_text.text_annotations
        text_list = [text.description for text in texts[1:]]
        combined_text = "\n".join(text_list)

    lines = combined_text.split("\n")
    for line in lines:
        words = line.split(" ")
        last_detected_vendor_sku = None  # Reset for each line

        if vendor_sku_entity_translator.matches_criteria(line):
            sku_number_data = line.replace("-", "").replace(" ", "").replace("SKU#", "")
            if len(sku_number_data) == 10 or len(sku_number_data) == 6:
                vendor_sku_word = sku_number_data
            else:
                vendor_sku_word = "".join(line.split(" ")[1:]).replace("-", "").replace(" ", "").replace("SKU#", "")
            last_detected_vendor_sku = vendor_sku_word
            if vendor_sku_word.strip() != "":
                print(f"[{file_name}] METRIC/VENDOR_SKU/{store_number}/{vendor_sku_word}/1")
                sku_entity_translator.translated_words.append(vendor_sku_word)

        for word in words:
            if sku_entity_translator.matches_criteria(word):
                # Extract only digits
                sku_word = re.sub(r"\D", "", word)
                # Skip if it matches the vendor SKU (to avoid duplication with vendor SKU)
                if last_detected_vendor_sku != sku_word and sku_word.strip() != "":
                    sku_entity_translator.translated_words.append(sku_word)
            else:
                # Try to extract candidate SKUs from OCR edge cases.
                candidate_skus = sku_entity_translator.extract_candidate_skus(word)
                for candidate_sku in candidate_skus:
                    if candidate_sku != last_detected_vendor_sku and candidate_sku.strip() != "":
                        sku_entity_translator.translated_words.append(candidate_sku)
                        recovered_skus.add(candidate_sku)
                        print(f'[{file_name}] Extracted candidate SKU from "{word}": {candidate_sku}')

    print(f"[{file_name}] Total SKUs found : {len(sku_entity_translator.translated_words)}")
    print(f"[{file_name}] {sku_entity_translator.translated_words}")
    return sku_entity_translator.translated_words, recovered_skus


# Pipeline order: Optional legacy fallback
# Description: Finds OCR bounding boxes corresponding to legacy SKU strings.
def detect_bounding_box(bounding_text, sku_list_collection, file_name=None):
    texts = bounding_text.text_annotations
    sku_list = []
    # Skip texts[0]: it is the full-page annotation whose bbox covers the entire image and
    # whose description is the concatenation of all page text. Matching against it would
    # produce bounding boxes that span the whole image (same reason find_sku_entities
    # uses texts[1:]).
    text_index = 1
    # Create a translator instance for candidate SKU extraction
    sku_translator = SKUEntityTranslator("skus")

    for text in texts:
        # Clean up: remove hyphens, spaces, and SKU prefixes (SKU:, SKU-, SKU#)
        text_value = str(text.description.replace("-", "")).replace(" ", "").replace("SKU:", "").replace("SKU-", "").replace("SKU#", "")
        if text_value in sku_list_collection:
            vertices = ["({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            sku_list.append([text_value, vertices])
        elif len(text_value) > 2:
            # Try to extract candidate SKUs from text that doesn't directly match
            candidates = sku_translator.extract_candidate_skus(text.description)
            matched_candidate = None
            for candidate in candidates:
                if candidate in sku_list_collection:
                    matched_candidate = candidate
                    break

            if matched_candidate:
                # Use the extracted SKU with original bounding box
                vertices = ["({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                sku_list.append([matched_candidate, vertices])
            elif text_index > 1 and text_index + 1 < len(texts):
                next_node = texts[text_index + 1]
                manipulated_text = str(
                    str((text.description + next_node.description).replace("-", ""))
                    .replace(" ", "")
                    .replace("SKU:", "")
                    .replace("SKU-", "")
                    .replace("SKU#", "")
                )
                if manipulated_text in sku_list_collection:
                    vertices = []
                    vertices.append("({},{})".format(text.bounding_poly.vertices[0].x, text.bounding_poly.vertices[0].y))
                    vertices.append("({},{})".format(next_node.bounding_poly.vertices[1].x, next_node.bounding_poly.vertices[1].y))
                    vertices.append("({},{})".format(next_node.bounding_poly.vertices[2].x, next_node.bounding_poly.vertices[2].y))
                    vertices.append("({},{})".format(text.bounding_poly.vertices[3].x, text.bounding_poly.vertices[3].y))
                    sku_list.append([manipulated_text, vertices])
                else:
                    prev_node = texts[text_index - 1]
                    manipulated_text = str(
                        str((prev_node.description + text.description + next_node.description).replace("-", ""))
                        .replace(" ", "")
                        .replace("SKU:", "")
                        .replace("SKU-", "")
                        .replace("SKU#", "")
                    )
                    if manipulated_text in sku_list_collection:
                        vertices = []
                        vertices.append("({},{})".format(prev_node.bounding_poly.vertices[0].x, prev_node.bounding_poly.vertices[0].y))
                        vertices.append("({},{})".format(next_node.bounding_poly.vertices[1].x, next_node.bounding_poly.vertices[1].y))
                        vertices.append("({},{})".format(next_node.bounding_poly.vertices[2].x, next_node.bounding_poly.vertices[2].y))
                        vertices.append("({},{})".format(prev_node.bounding_poly.vertices[3].x, prev_node.bounding_poly.vertices[3].y))
                        sku_list.append([manipulated_text, vertices])
        text_index = text_index + 1

    print(f"[{file_name}] sku_list: {sku_list}")
    return sku_list


# Pipeline order: Optional legacy fallback
# Description: Collects and formats all legacy OCR bounding boxes for one SKU.
def find_bounding_values(bounding_list, sku):
    bounding_text = []
    for row, sku_value in enumerate(bounding_list):
        try:
            column = sku_value.index(sku)
        except ValueError:
            continue
        bounding_text.append(structure_bounding(bounding_list[row][column + 1]))
    bounding_result = str(bounding_text).replace("[[", "{").replace("]]", "}").replace("[", "{").replace("]", "}")
    bounding_count = len(bounding_text)
    return bounding_result, bounding_count


# Pipeline order: Optional legacy fallback
# Description: Converts Google OCR polygon vertices into the legacy bbox coordinate order.
def structure_bounding(bounding_text):
    bounding_box = [
        bounding_text[3].split(",")[0].replace("(", ""),
        bounding_text[3].split(",")[1].replace(")", ""),
        bounding_text[1].split(",")[0].replace("(", ""),
        bounding_text[1].split(",")[1].replace(")", ""),
    ]
    return bounding_box
