"""Sentence templates grouped by writing condition."""

from __future__ import annotations


TEMPLATES: dict[str, tuple[str, ...]] = {
    "clean": (
        "Please contact {PERSON} at {EMAIL}.",
        "You can call {PERSON} on {PHONE_NUMBER} during office hours.",
        "The delivery address is {ADDRESS}.",
        "Please credit account {ACCOUNT_NUMBER} with {FINANCIAL_INFORMATION}.",
        "{PERSON} works at {ORGANIZATION} in {LOCATION}.",
        "Our {LOCATION} branch of {ORGANIZATION} can assist you.",
        "Send the receipt to {EMAIL} after transferring {FINANCIAL_INFORMATION}.",
        "The customer named {PERSON} provided account number {ACCOUNT_NUMBER}.",
        "For verification, use {PHONE_NUMBER} and {EMAIL}.",
        "A courier will collect the parcel from {ADDRESS} for {ORGANIZATION}.",
        "{PERSON} requested a meeting in {LOCATION}.",
        "The registered office of {ORGANIZATION} is at {ADDRESS}.",
        "Please send {FINANCIAL_INFORMATION} to account {ACCOUNT_NUMBER} for {PERSON}.",
        "Update the contact details to {PHONE_NUMBER} and {EMAIL}.",
        "The applicant {PERSON} currently lives at {ADDRESS}.",
        "Payment of {FINANCIAL_INFORMATION} was approved by {ORGANIZATION}.",
    ),
    "banglish": (
        "amar naam {PERSON}, phone number {PHONE_NUMBER}",
        "vai email ta {EMAIL} e pathai den",
        "amar bkash nmbr {PHONE_NUMBER}, taka pathai den",
        "{PERSON} er account {ACCOUNT_NUMBER} verify korben",
        "ami {LOCATION} e thaki, address {ADDRESS}",
        "{ORGANIZATION} er office {LOCATION} e ache",
        "{FINANCIAL_INFORMATION} payment ta {ACCOUNT_NUMBER} e diben",
        "apu {PERSON} ke {EMAIL} diye contact koren",
        "delivery ta {ADDRESS} e pathaben please",
        "amar fone {PHONE_NUMBER} ar mail {EMAIL}",
        "{PERSON} {ORGANIZATION} e kaj kore",
        "location {LOCATION}, okhane giye {PERSON} ke call diben",
        "{ACCOUNT_NUMBER} amar acc no, amount {FINANCIAL_INFORMATION}",
        "{ORGANIZATION} theke {PERSON} bolchi, nmbr {PHONE_NUMBER}",
        "amar bari {ADDRESS}, ami ekhon {LOCATION} e",
        "receipt ta {EMAIL} e diben for {FINANCIAL_INFORMATION}",
    ),
    "noisy": (
        "acc no {ACCOUNT_NUMBER} plz confirm asap",
        "hey its {PERSON} reach me {PHONE_NUMBER} or {EMAIL} thx",
        "send {FINANCIAL_INFORMATION} to {ACCOUNT_NUMBER} urgent",
        "addr {ADDRESS} deliver 2day",
        "{PERSON} frm {ORGANIZATION} call {PHONE_NUMBER}",
        "mail {EMAIL} dont call",
        "meet me {LOCATION} ask for {PERSON}",
        "new fone {PHONE_NUMBER} save it",
        "office moved {ADDRESS} update records",
        "wrkng at {ORGANIZATION} based in {LOCATION}",
        "{ACCOUNT_NUMBER} thats the acc nmbr",
        "invoice amt {FINANCIAL_INFORMATION} email {EMAIL}",
        "yo {PERSON} here msg me on {PHONE_NUMBER}",
        "payment {FINANCIAL_INFORMATION} for {ORGANIZATION} done",
        "ship to {ADDRESS} attn {PERSON}",
        "contact {ORGANIZATION} {LOCATION} branch asap",
    ),
}

VALID_GROUPS = tuple(TEMPLATES)
