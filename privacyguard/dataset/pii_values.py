"""Realistic but entirely fictional PII value generators."""

from __future__ import annotations

import random
import string
from collections.abc import Callable


ENGLISH_FIRST_NAMES = (
    "Aisha", "Daniel", "Emily", "Farhan", "Grace", "James", "Maria", "Noah",
    "Olivia", "Sofia", "Thomas", "William",
)
SOUTH_ASIAN_FIRST_NAMES = (
    "Afsana", "Anik", "Arif", "Farzana", "Hasan", "Ishrat", "Mahin", "Mehedi",
    "Nabila", "Nusrat", "Rafi", "Rumana", "Sadia", "Sakib", "Shamim", "Tanvir",
)
LAST_NAMES = (
    "Ahmed", "Akter", "Alam", "Chowdhury", "Das", "Haque", "Hossain", "Islam",
    "Khan", "Rahman", "Roy", "Sarker", "Sen", "Smith", "Williams",
)
ORGANIZATIONS = (
    "Alo Foundation", "Bengal Analytics", "Blue River Bank", "Dhaka Digital",
    "Eastern Trade Ltd", "Green Delta Services", "Meghna Finance", "Nabanna Foods",
    "Northstar Clinic", "Padma Telecom", "Shapla Technologies", "Sonali Mart",
)
LOCATIONS = (
    "Banani", "Barishal", "Chattogram", "Comilla", "Dhaka", "Dhanmondi",
    "Gulshan", "Khulna", "Mymensingh", "Rajshahi", "Rangpur", "Sylhet",
    "Uttara", "West Bengal",
)
STREETS = (
    "Lake Road", "Station Road", "Green Avenue", "Kazi Nazrul Road",
    "Shapla Lane", "College Road", "Park Street", "Begum Rokeya Avenue",
)
EMAIL_DOMAINS = ("example.com", "mail.test", "outlook.com", "demo.org", "sample.net")


class PIIValueGenerator:
    """Generate fictional entity values from a caller-supplied RNG."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def person(self, *, noisy_format: bool = False) -> str:
        first_pool = ENGLISH_FIRST_NAMES + SOUTH_ASIAN_FIRST_NAMES
        value = f"{self.rng.choice(first_pool)} {self.rng.choice(LAST_NAMES)}"
        return value.lower() if noisy_format and self.rng.random() < 0.45 else value

    def phone_number(self, *, noisy_format: bool = False) -> str:
        operator = self.rng.choice(("013", "014", "015", "016", "017", "018", "019"))
        digits = operator + "".join(self.rng.choices(string.digits, k=8))
        if not noisy_format:
            return digits
        style = self.rng.choice(("spaces", "dashes", "country_spaces", "country_dashes"))
        local = f"{digits[:3]} {digits[3:7]} {digits[7:]}" if "spaces" in style else f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
        return f"+88 {local}" if style == "country_spaces" else (f"+88-{local}" if style == "country_dashes" else local)

    def email(self, *, noisy_format: bool = False) -> str:
        first = self.rng.choice(ENGLISH_FIRST_NAMES + SOUTH_ASIAN_FIRST_NAMES).lower()
        last = self.rng.choice(LAST_NAMES).lower()
        separator = self.rng.choice((".", "_", ""))
        suffix = str(self.rng.randint(1, 999)) if self.rng.random() < 0.65 else ""
        return f"{first}{separator}{last}{suffix}@{self.rng.choice(EMAIL_DOMAINS)}"

    def address(self, *, noisy_format: bool = False) -> str:
        house = self.rng.randint(1, 299)
        street = self.rng.choice(STREETS)
        area = self.rng.choice(LOCATIONS)
        if noisy_format:
            return self.rng.choice((f"hse {house} {street.lower()} {area.lower()}", f"{house}/{self.rng.randint(1, 20)} {street} {area}"))
        return f"House {house}, {street}, {area}"

    def account_number(self, *, noisy_format: bool = False) -> str:
        digits = "".join(self.rng.choices(string.digits, k=self.rng.choice((10, 12, 13))))
        if not noisy_format:
            return digits
        separator = self.rng.choice((" ", "-"))
        return separator.join(digits[index:index + 4] for index in range(0, len(digits), 4))

    def financial_information(self, *, noisy_format: bool = False) -> str:
        amount = self.rng.randint(5, 950) * 100
        if noisy_format:
            return self.rng.choice((f"tk {amount}", f"৳{amount:,}", f"bdt {amount:,}"))
        return self.rng.choice((f"BDT {amount:,}", f"Tk {amount:,}", f"USD {self.rng.randint(10, 900):,}"))

    def organization(self, *, noisy_format: bool = False) -> str:
        value = self.rng.choice(ORGANIZATIONS)
        return value.lower() if noisy_format and self.rng.random() < 0.55 else value

    def location(self, *, noisy_format: bool = False) -> str:
        value = self.rng.choice(LOCATIONS)
        return value.lower() if noisy_format and self.rng.random() < 0.55 else value

    def generate(self, label: str, *, noisy_format: bool = False) -> str:
        generators: dict[str, Callable[..., str]] = {
            "PERSON": self.person,
            "PHONE_NUMBER": self.phone_number,
            "EMAIL": self.email,
            "ADDRESS": self.address,
            "ACCOUNT_NUMBER": self.account_number,
            "FINANCIAL_INFORMATION": self.financial_information,
            "ORGANIZATION": self.organization,
            "LOCATION": self.location,
        }
        try:
            generator = generators[label]
        except KeyError as error:
            raise ValueError(f"unknown entity label: {label}") from error
        return generator(noisy_format=noisy_format)
