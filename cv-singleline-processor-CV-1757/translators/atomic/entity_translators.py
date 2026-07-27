from abc import ABC, abstractmethod
import re


# Pipeline order: Optional legacy fallback
# Description: Base class for regex-based OCR entity translators.
class BaseEntityTranslator(ABC):

    # Pipeline order: Optional legacy fallback
    # Description: Stores the translator name and collected translated words.
    def __init__(self, name):
        self.name = name
        self.translated_words = []

    @abstractmethod
    # Pipeline order: Optional legacy fallback
    # Description: Requires subclasses to provide regex patterns used for matching.
    def get_matching_criteria(self):
        pass

    @abstractmethod
    # Pipeline order: Optional legacy fallback
    # Description: Checks whether a word matches any regex pattern for this entity type.
    def matches_criteria(self, word):
        word_matched = False
        for criteria in self.get_matching_criteria():
            match = re.search(criteria, word)
            if match:
                word_matched = True
                break
        return word_matched


# Pipeline order: Optional legacy fallback
# Description: Regex translator for SKU and vendor-SKU text extracted from OCR.
class SKUEntityTranslator(BaseEntityTranslator):

    # Pipeline order: Optional legacy fallback
    # Description: Creates a SKU or vendor-SKU translator instance.
    def __init__(self, name):
        super().__init__(name)

    # Pipeline order: Optional legacy fallback
    # Description: Returns regex rules for valid SKU and vendor-SKU OCR formats.
    def get_matching_criteria(self):
        # TODO: is it better to come up with one regex that can capture all possibilities?
        # EX: This (?:^(?:\d{6})$)|(?:^(?:\d{10})$)|(?:^(?:\d{3}-\d{3})$)|(?:^(?:\d{4}-\d{3}-\d{3})$) combines
        # all the 4 below. Doing so removes the need to iterate over each, though less readable.
        matching_criteria = [
            r"^(?:\d{6})$",  # non hyphenated 6 digits SKU
            r"^(?:\d{10})$",  # non hyphenated 10 digits SKU
            r"^(?:\d{3}-\d{3})$",  # hyphenated 6 digits SKU
            r"^(?:\d{4}-\d{3}-\d{3})$",  # hyphenated 10 digits SKU
            r"^(?:SKU:\d{6})$",  # SKU: prefix with 6 digits
            r"^(?:SKU:\d{10})$",  # SKU: prefix with 10 digits
            r"^(?:SKU-\d{6})$",  # SKU- prefix with 6 digits
            r"^(?:SKU-\d{10})$"  # SKU- prefix with 10 digits
        ]
        if self.name == 'vendor_skus':
            matching_criteria = [
                r"^(?:\d{4}\W\d{3}\W\d{3})$",  # white space or hyphenated 2 digits Other + 10 digits SKU
                r"^(?:\d{2}\W\d{4}\W\d{3}\W\d{3})$",  # white space or hyphenated 2 digits Other + 10 digits SKU
                r"^(?:\d{2}\WSKU#\d{4}\W\d{3}\W\d{3})$",
                # white space or hyphenated 2 digits Other + SKU# Text + 10 digits SKU
                r"^(?:SKU#\d{4}\W\d{3}\W\d{3})$",  # white space or hyphenated SKU# Text + 10 digits SKU
                r"^(?:SKU\W\d{4}\W\d{3}\W\d{3})$",  # white space or hyphenated SKU Text + 10 digits SKU
                r"^(?:SKU#\d{10})$",  # white space or hyphenated SKU# Text + 10 digits SKU
                r"^(?:\d{6})$",  # white space or hyphenated 2 digits Other + 6 digits SKU
                r"^(?:\d{2}\W\d{6})$",  # white space or hyphenated 2 digits Other + 6 digits SKU
                r"^(?:\d{2}\WSKU\W#\d{6})$",  # white space or hyphenated 2 digits Other + SKU # Text + 6 digits SKU
                r"^(?:\d{2}\WSKU#\W\d{6})$",  # white space or hyphenated 2 digits Other + SKU# Text + 6 digits SKU
                r"^(?:\d{3}\W\d{3})$",  # white space or hyphenated 2 digits Other + (3+3) digits SKU
                r"^(?:\d{2}\W\d{3}\W\d{3})$",  # white space or hyphenated 2 digits Other + (3+3) digits SKU
                r"^(?:\d{2}\W\d{10})$",  # hyphenated 2 digits Other + 10 digits SKU
            ]
        return matching_criteria

    # Pipeline order: Optional legacy fallback
    # Description: Applies SKU regex matching to one OCR word.
    def matches_criteria(self, word):
        return super().matches_criteria(word)

    # Pipeline order: Optional legacy fallback
    # Description: Recovers plausible SKUs from noisy OCR strings that do not exactly match base regexes.
    def extract_candidate_skus(self, text):
        """
        Extract potential SKUs from text that don't exactly match the regex patterns.
        This handles common OCR edge cases:
        - 11 chars with "10" at position 1-2: extract the 10-digit SKU starting from position 1
        - 7 digits starting with "1" but NOT "10" at position 0-1: extract the 6-digit SKU from position 1
        - 11 or 12 digits starting with "10": extract the first 10 digits
        - Length > 10: scan for all 10-digit substrings starting with "10"
        
        Args:
            text: The text to extract candidate SKUs from
            
        Returns:
            List of candidate SKU strings that passed validation
        """
        candidates = []
        
        # Skip date patterns like MM/DD/YY, DD/MM/YY, MM-DD-YY, etc.
        date_patterns = [
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # MM/DD/YY or MM/DD/YYYY or D/M/YY
            r'^\d{1,2}-\d{1,2}-\d{2,4}$',  # MM-DD-YY or MM-DD-YYYY
            r'^\d{4}-\d{1,2}-\d{1,2}$',    # YYYY-MM-DD
        ]
        for pattern in date_patterns:
            if re.match(pattern, text):
                return candidates  # Return empty list for date patterns
        
        # Remove all non-digit characters (including colons, spaces, hyphens, etc.)
        text_clean = re.sub(r'\D', '', text)
        if len(text_clean) == 10 and text_clean[0:2] == '10' and self.matches_criteria(text_clean):
            candidates.append(text_clean)
        # Case 1: 11 characters with "10" at position 1-2, extract 10-digit SKU
        if len(text_clean) == 11 and text_clean[1:3] == '10':
            candidate = text_clean[1:11]  # Extract 10 digits starting from position 1
            if self.matches_criteria(candidate):
                candidates.append(candidate)
        
        # # Case 2: 7 digits starting with "1", but NOT "10" at position 0-1, extract 6-digit SKU
        # if len(text_clean) == 7 and text_clean[0] == '1' and text_clean[1] != '0':
        #     candidate = text_clean[1:7]  # Extract 6 digits from position 1
        #     if self.matches_criteria(candidate):
        #         candidates.append(candidate)
        
        # Case 3: 11 or 12 digits starting with "10", extract the first 10 digits
        if len(text_clean) in [11, 12] and text_clean[0:2] == '10':
            candidate = text_clean[0:10]  # Extract first 10 digits
            if self.matches_criteria(candidate):
                candidates.append(candidate)
        
        # Case 4: Length > 10, extract last 10 digits if they start with "10"
        # This handles cases like "22 1004215338" or "-338 1004215338" → "1004215338"
        if len(text_clean) > 10:
            last_10 = text_clean[-10:]
            if last_10[0:2] == '10':
                if self.matches_criteria(last_10) and last_10 not in candidates:
                    candidates.append(last_10)
        
        return candidates