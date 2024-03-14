---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Download my CV in [PDF](../files/Ankit_cv.pdf)

Education
======
* Ph.D in Physics, Indian Institute of Science Education and Research (IISER), Mohali, India, 2014-2020
  - Area: Galaxy evolution, large-scale structures
  - Supervisor: Prof. Jasjeet Singh Bagla
* M.Sc. in Physics, University of Delhi, India, 2012-2014
* B.Sc. in Physics, University of Delhi, India, 2009-2012

Work experience
======
* 2020-Present: Research Fellow
  * Korea Institute for Advanced Study (KIAS), Seoul, Republic of Korea
  * Area: Galaxy evolution, large-scale structures, cosmological hydrodynamical simulations
  * Mentor: Prof. Changbom Park


Skills
======
* Python, C, Fortran
* High performance computing
* Data visualization
* Data cleaning

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  

Academic Outreach Activities
======
 <ul>{% for post in site.outreach reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Organization
======
 <ul>{% for post in site.organization reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
