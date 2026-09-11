from __future__ import annotations

import unittest

from privacyguard.models.bio import categories_to_bio


class BIOReconstructionTests(unittest.TestCase):
    def test_contiguous_categories_use_inside_tags(self) -> None:
        categories = [
            "PERSON",
            "PERSON",
            "O",
            "PERSON",
            "EMAIL",
            "EMAIL",
            "PHONE_NUMBER",
            "PHONE_NUMBER",
        ]
        self.assertEqual(
            categories_to_bio(categories),
            [
                "B-PERSON",
                "I-PERSON",
                "O",
                "B-PERSON",
                "B-EMAIL",
                "I-EMAIL",
                "B-PHONE_NUMBER",
                "I-PHONE_NUMBER",
            ],
        )


if __name__ == "__main__":
    unittest.main()
