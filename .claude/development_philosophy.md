# PlasmaDX Development Philosophy

## Core Principles

### 1. Quality Over Speed
- **Never compromise on technical soundness to save time**
- Time is NOT a factor in decision-making
- Always choose the most robust, future-proof solution
- Complexity is acceptable if it's the right solution

### 2. Technical Excellence
- Prioritize: Efficiency, Reliability, Extensibility, Maintainability
- Avoid cutting corners or "stopgap" solutions
- Think long-term: "Will this cause problems down the road?"
- If a proper solution takes longer, that's perfectly fine

### 3. Work Schedule
- Developer works throughout the night (normal schedule)
- No need to rush or compress timelines
- Take the time needed to do it right the first time

### 4. Decision Making Process
When faced with multiple approaches:
1. Identify the most technically sound solution
2. Evaluate long-term implications
3. Choose based on technical merit, NOT time savings
4. Document why this approach is superior

### 5. Problem Solving Approach
- Use agents to thoroughly assess complex problems
- Research best practices and industry standards
- Formulate comprehensive plans before implementation
- Don't settle for incremental improvements when fundamental fixes are needed

## Application to Current Work

### Color Depth Issue Example
- **NOT acceptable:** "Let's use 10-bit for now, do 16-bit later"
- **Acceptable:** "16-bit HDR with proper blit pipeline is the right solution, let's implement it fully"
- **Reasoning:** The violent flashing/stuttering is a critical quality issue that needs proper resolution

### Key Insight
Multiple issues may contribute to a problem (color depth + other factors). Address each properly rather than accepting partial solutions.

## Agent Deployment
When uncertain or facing complex decisions:
1. Deploy multiple agents to assess the situation
2. Research competing approaches
3. Formulate detailed implementation plans
4. Execute the best solution comprehensively

---

**Last Updated:** 2025-10-14
**Purpose:** Ensure all Claude sessions follow consistent development principles for PlasmaDX
