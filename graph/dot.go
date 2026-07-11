package graph

import (
	"fmt"
	"strings"
)

// ToDOT returns a Graphviz DOT representation of the state graph.
// The output can be rendered with `dot -Tpng graph.dot -o graph.png`.
//
//	dot := g.ToDOT()
//	_ = os.WriteFile("graph.dot", []byte(dot), 0644)
func (g *StateGraph[S]) ToDOT() string {
	var sb strings.Builder
	sb.WriteString("digraph ")
	if g.name != "" {
		sb.WriteString(g.name)
	} else {
		sb.WriteString("StateGraph")
	}
	sb.WriteString(" {\n  rankdir=LR;\n  node [shape=box, style=rounded];\n\n")

	// START / END are styled differently
	sb.WriteString("  __start__ [label=\"START\", shape=circle, style=filled, fillcolor=\"#e0f0e0\"];\n")
	sb.WriteString("  __end__   [label=\"END\", shape=doublecircle, style=filled, fillcolor=\"#f0e0e0\"];\n\n")

	// Nodes
	for name := range g.nodes {
		label := escapeDOT(name)
		fmt.Fprintf(&sb, "  %q [label=%q];\n", name, label)
	}
	sb.WriteString("\n")

	// Edges
	for from, edges := range g.edges {
		for _, e := range edges {
			attrs := ""
			if e.interrupt {
				attrs = " [style=dashed, color=red, label=\"⏸\"]"
			}
			switch e.kind {
			case edgeUnconditional:
				fmt.Fprintf(&sb, "  %q -> %q%s;\n", from, e.to, attrs)
			case edgeParallel:
				for _, branch := range e.branches {
					fmt.Fprintf(&sb, "  %q -> %q [color=blue, label=\"∥\"];\n", from, branch)
				}
			case edgeConditional:
				for key, to := range e.mapping {
					label := escapeDOT(key)
					fmt.Fprintf(&sb, "  %q -> %q [label=%q];\n", from, to, label)
				}
			}
		}
	}

	sb.WriteString("}\n")
	return sb.String()
}

// ToDOT returns a DOT representation for a CompiledGraph.
func (c *CompiledGraph[S]) ToDOT() string {
	var sb strings.Builder
	name := c.name
	if name == "" {
		name = "CompiledGraph"
	}
	sb.WriteString("digraph ")
	sb.WriteString(name)
	sb.WriteString(" {\n  rankdir=LR;\n  node [shape=box, style=rounded];\n\n")

	sb.WriteString("  __start__ [label=\"START\", shape=circle, style=filled, fillcolor=\"#e0f0e0\"];\n")
	sb.WriteString("  __end__   [label=\"END\", shape=doublecircle, style=filled, fillcolor=\"#f0e0e0\"];\n\n")

	for name := range c.nodes {
		label := escapeDOT(name)
		fmt.Fprintf(&sb, "  %q [label=%q];\n", name, label)
	}
	sb.WriteString("\n")

	for from, edges := range c.edges {
		for _, e := range edges {
			attrs := ""
			if e.interrupt {
				attrs = " [style=dashed, color=red, label=\"⏸\"]"
			}
			switch e.kind {
			case edgeUnconditional:
				fmt.Fprintf(&sb, "  %q -> %q%s;\n", from, e.to, attrs)
			case edgeParallel:
				for _, branch := range e.branches {
					fmt.Fprintf(&sb, "  %q -> %q [color=blue, label=\"∥\"];\n", from, branch)
				}
			case edgeConditional:
				for key, to := range e.mapping {
					label := escapeDOT(key)
					fmt.Fprintf(&sb, "  %q -> %q [label=%q];\n", from, to, label)
				}
			}
		}
	}

	sb.WriteString("}\n")
	return sb.String()
}

func escapeDOT(s string) string {
	s = strings.ReplaceAll(s, `"`, `\"`)
	s = strings.ReplaceAll(s, "\n", "\\n")
	return s
}
