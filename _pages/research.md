---
layout: archive
title: "Research"
permalink: /research/
author_profile: true
---


{% include base_path %}

{% for post in site.code reversed %}  
  {% if post.collection == 'research' %}  
    {% include archive-single.html %}  
  {% endif %}  
{% endfor %}  