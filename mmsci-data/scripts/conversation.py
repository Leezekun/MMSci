import random
from utils import *

class Conversation:
    def __init__(self):
    
        self.detailed_caption_templates = [
            "Can you describe the entire figure including all sub-figures based on the article content?\n{}: {}",
            "Detail the entire figure including all sub-figures as per the article content.\n{}: {}",
            "What content is shown in the whole figure including all sub-figures?\n{}: {}",
            "What is depicted in the figure including all sub-figures according to the article?\n{}: {}",
            "Describe all elements in the figure including all sub-figures from the article.\n{}: {}",
            "Summarize the entire figure including all sub-figures based on the article.\n{}: {}",
            "Detail the information in the figure including all sub-figures using the article.\n{}: {}",
            "Describe the whole figure including all sub-figures from the article content.\n{}: {}",
            "Explain everything in the figure including all sub-figures as per the article.\n{}: {}",
            "What do the figure and all sub-figures illustrate according to the article?\n{}: {}",
            "Summarize the contents of the figure including all sub-figures based on the article.\n{}: {}",
            "What are the main points of the figure including all sub-figures from the article?\n{}: {}",
            "What is the summary of the figure including all sub-figures from the article?\n{}: {}",
            "Provide an in-depth summary of the figure including all sub-figures using the article.\n{}: {}",
            "What information is conveyed by the figure including all sub-figures as per the article?\n{}: {}",
            "Give an outline of the figure including all sub-figures based on the article.\n{}: {}",
            "What do the figure and all sub-figures represent according to the article?\n{}: {}",
            "What are the key details in the figure including all sub-figures from the article?\n{}: {}",
            "Summarize the figure including all sub-figures as per the article.\n{}: {}",
            "What is the primary message of the figure including all sub-figures from the article?\n{}: {}",
            "Explain the figure including all sub-figures in detail using the article.\n{}: {}",
            "What message do the figure and all sub-figures communicate based on the article?\n{}: {}",
            "What is the figure and all sub-figures' subject matter from the article?\n{}: {}",
            "Summarize the main aspects of the figure including all sub-figures using the article.\n{}: {}",
            "What do the figure and all sub-figures reveal as described in the article?\n{}: {}",
            "What is the complete summary of the figure including all sub-figures from the article?\n{}: {}",
            "Give an overall description of the figure including all sub-figures using the article.\n{}: {}"
        ]

        self.detailed_subcaption_templates = [
            "Can you describe the content in sub-figure ({})?",
            "Would you please describe the content in sub-figure ({})?",
            "What about the content in sub-figure ({})?",
            "Could you tell me the details in sub-figure ({})?",
            "What information is conveyed in sub-figure ({})?",
            "Summarize sub-figure ({}).",
            "Tell me about the content in sub-figure ({})",
            "What does sub-figure ({}) show?",
            "What can you tell me about the content in sub-figure ({})?",
            "What about sub-figure ({})?",
            "Could you explain what is shown in sub-figure ({})?",
            "Please summarize the content in sub-figure ({})",
            "Describe the data presented in sub-figure ({})",
            "What details are included in sub-figure ({})?",
            "Provide an overview of the information in sub-figure ({})",
            "What does the content in sub-figure ({}) depict?",
            "How would you summarize sub-figure ({})?",
            "What can be observed in sub-figure ({})?",
            "Give a detailed description of sub-figure ({})",
            "What key points are illustrated in sub-figure ({})?",
            "Can you provide a breakdown of sub-figure ({})?",
            "What insights are provided by sub-figure ({})?",
            "What is the main information in sub-figure ({})?",
            "Could you give an explanation of sub-figure ({})?",
            "What analysis can be made from sub-figure ({})?",
            "Describe what is presented in sub-figure ({})",
            "Can you detail the main points in sub-figure ({})?",
            "What is shown in sub-figure ({})?",
            "Give a brief description of sub-figure ({})",
            "What content is sub-figure ({}) illustrating?",
            "Could you discuss the details shown in sub-figure ({})?",
            "Explain the data in sub-figure ({})",
            "What summary can you provide for sub-figure ({})?",
            "What does sub-figure ({}) represent?",
            "What can you infer from sub-figure ({})?",
            "Give an explanation of what sub-figure ({}) shows",
            "Tell me the information depicted in sub-figure ({})",
            "What are the details conveyed by sub-figure ({})?",
            "Provide a summary of sub-figure ({})",
            "What findings are shown in sub-figure ({})?",
            "What does the diagram in sub-figure ({}) illustrate?",
            "Can you interpret the data in sub-figure ({})?",
            "What is depicted in sub-figure ({})?",
            "Please explain the findings in sub-figure ({})",
            "What are the main observations in sub-figure ({})?",
            "What does sub-figure ({}) indicate?",
            "How would you interpret sub-figure ({})?",
            "Describe the results shown in sub-figure ({})",
            "What is the key message of sub-figure ({})?",
            "What conclusions can be drawn from sub-figure ({})?",
            "What does sub-figure ({}) demonstrate?",
            "Can you give an overview of sub-figure ({})?",
            "What are the primary details in sub-figure ({})?",
            "What is illustrated in sub-figure ({})?",
            "Can you provide insights on sub-figure ({})?",
            "Please explain what is presented in sub-figure ({})",
            "What can be concluded from sub-figure ({})?",
            "What observations can be made from sub-figure ({})?",
            "Describe the key elements in sub-figure ({})",
            "What does sub-figure ({}) reveal?",
            "Summarize the main points of sub-figure ({})",
            "What can be discerned from sub-figure ({})?",
            "How would you explain sub-figure ({})?",
            "What are the essential points in sub-figure ({})?",
            "What is the focus of sub-figure ({})?",
            "Provide a detailed analysis of sub-figure ({})",
            "What main points are shown in sub-figure ({})?",
            "Can you break down the information in sub-figure ({})?",
            "What does sub-figure ({}) inform us about?",
            "What is the takeaway from sub-figure ({})?"
        ]

        self.concise_follow_up_templates = [
            "What about sub-figure ({})?",
            "And sub-figure ({})?",
            "sub-figure ({})?",
            "Describe sub-figure ({}).",
            "Summarize sub-figure ({}).",
            "Explain sub-figure ({}).",
            "What's in sub-figure ({})?",
            "Content of sub-figure ({})?",
            "Details in sub-figure ({})?",
            "Data in sub-figure ({})?",
            "What is sub-figure ({})?",
            "Insights on sub-figure ({})?",
            "Overview of sub-figure ({})?",
            "Key points in sub-figure ({})?",
            "Main info in sub-figure ({})?",
            "What does sub-figure ({}) say?",
            "Summary of sub-figure ({})?",
            "Explain the data in sub-figure ({})?",
            "Breakdown of sub-figure ({})?",
            "Analyze sub-figure ({}).",
            "Explain findings in sub-figure ({}).",
            "Main details of sub-figure ({}).",
            "Discuss sub-figure ({}).",
            "Review sub-figure ({}).",
            "What's depicted in sub-figure ({})?",
            "Highlight of sub-figure ({})?",
            "What’s shown in sub-figure ({})?",
            "Explain content of sub-figure ({})?",
            "Content summary of sub-figure ({})?",
            "Interpret sub-figure ({})."
        ]

        # extend the templates
        self.detailed_caption_templates = self.extend_templates(self.detailed_caption_templates)
        self.detailed_subcaption_templates = self.extend_templates(self.detailed_subcaption_templates)
        self.concise_follow_up_templates = self.extend_templates(self.concise_follow_up_templates)

    def extend_templates(self, conversation_templates):
        # Replacements for diversity
        replacements = [
            ("Can you", "Could you"),
            ("Can you", "Would you"),
            ("Can you", "Please"),
            ("describe", "explain"),
            ("describe", "detail"),
            ("describe", "summarize"),
            ("content", "information"),
            ("content", "details"),
            ("information", "details"),
            ("Tell me", "Explain to me"),
            ("Tell me", "Give me details about"),
            ("Tell me", "Provide an overview of"),
            ("Tell me", "Describe"),
            ("Tell me", "Summarize"),
            ("content", "data"),
            ("content", "information"),
            ("entire figure", "whole figure"),
        ]

        new_templates = []

        for template in conversation_templates:
            new_templates.append(template)
            for src, target in replacements:
                if src in template:
                    new_templates.append(template.replace(src, target))
                elif target in template:
                    new_templates.append(template.replace(target, src))

        conversation_templates = remove_duplicates(new_templates)
        return conversation_templates

    def get_detailed_caption(self):
        # Get the detailed discussion of the whole figure
        return random.choice(self.detailed_caption_templates)

    def get_conv_template(self, size, concise_ratio=0.5):
        conversation = []
        concise_needed = int(size * concise_ratio)
        full_needed = size - concise_needed

        # Ensure the first sentence is not from concise_follow_up_templates
        conversation.append(random.choice(self.detailed_subcaption_templates))
        full_needed -= 1

        # Use conversation templates first
        conversation.extend(random.sample(self.detailed_subcaption_templates, full_needed))

        # Use concise follow-up templates
        conversation.extend(random.sample(self.concise_follow_up_templates, concise_needed))

        # Shuffle to simulate conversation flow, except the first element
        random.shuffle(conversation[1:])

        return conversation
