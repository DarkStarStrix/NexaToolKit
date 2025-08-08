package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"
)

const version = "1.0.0"
const backendURL = "http://127.0.0.1:8000"
const licenseEnvVar = "NEXA_LICENSE_KEY"

// --- Styles & Art ---

var splashColors = []lipgloss.Color{
	lipgloss.Color("#39FF14"), // Green
	lipgloss.Color("#00FFFF"), // Cyan
	lipgloss.Color("#FF00FF"), // Magenta
	lipgloss.Color("#FFD700"), // Gold
	lipgloss.Color("#6C63FF"), // Purple
}

var (
	boxStyle = lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			Padding(1, 2).
			BorderForeground(lipgloss.Color("#6C63FF"))

	headerStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#6C63FF")).
			Padding(0, 1)

	selectedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFD21F")).
			Bold(true)

	successStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("42"))

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("196"))

	infoStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("69"))

	footerStyle = lipgloss.NewStyle().Faint(true).Align(lipgloss.Right)
	helpStyle   = lipgloss.NewStyle().Padding(0, 1)
)

// --- Data Structures ---

// BackendRequest defines the structure for requests sent to the Python backend.
type BackendRequest struct {
	Operation string                 `json:"operation"`
	FilePath  string                 `json:"file_path"`
	FilePaths []string               `json:"file_paths,omitempty"` // For merging
	Options   map[string]interface{} `json:"options"`
}

// BackendResponse defines the structure for responses from the Python backend.
type BackendResponse struct {
	Success       bool        `json:"success"`
	Message       string      `json:"message"`
	Data          interface{} `json:"data"`
	ExecutionTime float64     `json:"execution_time"`
	Error         string      `json:"error"`
}

// ErrorDetail is used to decode nested error details from FastAPI's HTTPException.
type ErrorDetail struct {
	Detail BackendResponse `json:"detail"`
}

// ModelInfo represents model metadata returned by the backend.
type ModelInfo struct {
	FormatType    string       `json:"format_type"`
	FileSize      int64        `json:"file_size"`
	MemorySize    int64        `json:"memory_size"`
	NumParameters int64        `json:"num_parameters"`
	NumTensors    int          `json:"num_tensors"`
	TensorInfo    []TensorInfo `json:"tensor_info"`
	Architecture  string       `json:"architecture"`
	Dtype         string       `json:"dtype"`
}

// TensorInfo represents metadata for a single tensor in a model.
type TensorInfo struct {
	Name  string `json:"name"`
	Shape []int  `json:"shape"`
	Dtype string `json:"dtype"`
	Size  int64  `json:"size"`
}

// Operation represents an available action in the TUI.
type Operation struct {
	Name        string
	Description string
	Command     string
}

// operationItem wraps an Operation to implement the list.Item interface for Bubble Tea.
type operationItem struct {
	op Operation
}

// Available operations for the TUI and CLI.
var operations = []Operation{
	{
		Name:        "Analyze Model",
		Description: "Extract model statistics and metadata",
		Command:     "analyze",
	},
	{
		Name:        "Convert Format",
		Description: "Convert between PyTorch and SafeTensors formats",
		Command:     "convert",
	},
	{
		Name:        "Quantize Model",
		Description: "Reduce model size through quantization",
		Command:     "quantize",
	},
	{
		Name:        "Prune Model",
		Description: "Reduce parameters via weight pruning (requires full model)",
		Command:     "prune",
	},
	{
		Name:        "Merge Models",
		Description: "Merge the weights of two or more models",
		Command:     "merge",
	},
	{
		Name:        "Benchmark Performance",
		Description: "Test model inference latency and throughput",
		Command:     "benchmark",
	},
	{
		Name:        "Run Evaluation",
		Description: "Run a custom Python script to evaluate the model",
		Command:     "eval",
	},
	{
		Name:        "Check Environment",
		Description: "Validate dependencies and system resources",
		Command:     "check",
	},
}

// TUI model states.
type state int

const (
	stateTextInput state = iota
	stateMenu
	stateProcessing
	stateResult
	stateError
	stateLicense
	stateExtraInput
)

// model is the main Bubble Tea model for the TUI.
type model struct {
	state               state
	opList              list.Model
	textInput           textinput.Model
	spinner             spinner.Model
	progress            progress.Model
	filePath            string
	filePaths           []string // For merge
	result              *BackendResponse
	err                 error
	selectedOp          Operation
	splash              string
	width               int
	height              int
	licenseKey          string
	evalScriptPath      string // For eval
	benchmarkInputShape string // For benchmark
}

// --- TUI Messages ---

// backendMsg is a message sent when a backend operation completes.
type backendMsg struct {
	response *BackendResponse
	err      error
}

// progressMsg is a message for updating progress bars (currently unused).
type progressMsg float64

// --- TUI Implementation ---

// initialModel creates the initial state of the TUI model.
func initialModel(filePath string) model {
	opItems := make([]list.Item, len(operations))
	for i, op := range operations {
		opItems[i] = operationItem{op: op}
	}

	delegate := list.NewDefaultDelegate()
	delegate.Styles.SelectedTitle = selectedStyle.Copy().Padding(0, 1)
	delegate.Styles.SelectedDesc = selectedStyle.Copy().Faint(true).Padding(0, 1)

	opList := list.New(opItems, delegate, 0, 0)
	opList.Title = "Nexa Toolkit - Select Operation"
	opList.SetShowStatusBar(false)
	opList.SetFilteringEnabled(false)

	ti := textinput.New()
	ti.Placeholder = "C:\\path\\to\\your\\model.pt"
	ti.Focus()
	ti.CharLimit = 256
	ti.Width = 80

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	p := progress.New(progress.WithDefaultGradient())

	initialState := stateMenu
	if filePath == "" {
		initialState = stateTextInput
	}

	// Check for license key
	licenseKey := os.Getenv(licenseEnvVar)
	if licenseKey == "" {
		initialState = stateLicense
		ti.Placeholder = "Enter your license key"
	}

	return model{
		state:               initialState,
		opList:              opList,
		textInput:           ti,
		spinner:             s,
		progress:            p,
		filePath:            filePath,
		splash:              CreateNexaSplash(),
		licenseKey:          licenseKey,
		benchmarkInputShape: "1,3,224,224", // Default value
	}
}

func (i operationItem) Title() string       { return i.op.Name }
func (i operationItem) Description() string { return i.op.Description }
func (i operationItem) FilterValue() string { return i.op.Name }

// Init initializes the TUI model.
func (m model) Init() tea.Cmd {
	if m.state == stateTextInput || m.state == stateLicense {
		return textinput.Blink
	}
	return m.spinner.Tick
}

// Update handles messages and updates the TUI model.
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Global quit
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}

		switch m.state {
		case stateLicense:
			switch msg.String() {
			case "enter":
				key := m.textInput.Value()
				if key != "" {
					m.licenseKey = key
					m.state = stateTextInput
					m.textInput.SetValue("")
					m.textInput.Placeholder = "C:\\path\\to\\your\\model.pt"
					m.err = nil
				} else {
					m.err = fmt.Errorf("license key cannot be empty")
				}
				return m, nil
			}
			m.textInput, cmd = m.textInput.Update(msg)
			return m, cmd

		case stateTextInput:
			switch msg.String() {
			case "q", "esc":
				return m, tea.Quit
			case "enter":
				path := m.textInput.Value()
				if isValidModelFile(path) {
					m.filePath = path
					m.state = stateMenu
					m.err = nil // Clear previous error
					return m, nil
				}
				m.err = fmt.Errorf("invalid or non-existent file path: %s", path)
				return m, nil
			}
			m.textInput, cmd = m.textInput.Update(msg)
			return m, cmd

		case stateMenu:
			switch msg.String() {
			case "q":
				return m, tea.Quit
			case "enter":
				if item, ok := m.opList.SelectedItem().(operationItem); ok {
					m.selectedOp = item.op

					// Check for operations needing extra input
					switch item.op.Command {
					case "merge":
						m.state = stateExtraInput
						m.textInput.Placeholder = "Enter comma-separated paths to models"
						m.textInput.SetValue("")
						m.textInput.Focus()
						return m, textinput.Blink
					case "benchmark":
						m.state = stateExtraInput
						m.textInput.Placeholder = "Input shape (e.g., 1,3,224,224)"
						m.textInput.SetValue(m.benchmarkInputShape) // Default value
						m.textInput.Focus()
						return m, textinput.Blink
					case "eval":
						m.state = stateExtraInput
						m.textInput.Placeholder = "Path to evaluation script (.py)"
						m.textInput.SetValue("")
						m.textInput.Focus()
						return m, textinput.Blink
					}

					// For other operations
					if item.op.Command != "check" && m.filePath == "" {
						m.state = stateError
						m.err = fmt.Errorf("this operation requires a model file. Please restart and provide a file path")
						return m, nil
					}

					m.state = stateProcessing
					return m, tea.Batch(
						m.spinner.Tick,
						m.runBackendOperation(item.op.Command),
					)
				}
			}
		case stateResult, stateError:
			switch msg.String() {
			case "b", "enter", "esc", "q":
				m.state = stateMenu
				m.result = nil
				m.err = nil
				return m, nil
			}
		case stateExtraInput:
			switch msg.String() {
			case "q", "esc":
				m.state = stateMenu
				m.textInput.SetValue("")
				m.textInput.Blur()
				return m, nil
			case "enter":
				value := m.textInput.Value()
				if value == "" {
					m.err = fmt.Errorf("input cannot be empty")
					return m, nil
				}
				m.err = nil

				switch m.selectedOp.Command {
				case "merge":
					m.filePaths = strings.Split(value, ",")
				case "benchmark":
					m.benchmarkInputShape = value
				case "eval":
					m.evalScriptPath = value
				}

				m.state = stateProcessing
				return m, tea.Batch(
					m.spinner.Tick,
					m.runBackendOperation(m.selectedOp.Command),
				)
			}
			m.textInput, cmd = m.textInput.Update(msg)
			return m, cmd
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.opList.SetWidth(msg.Width)
		m.opList.SetHeight(msg.Height - 2)
		// Set the width of the text input, accounting for box padding.
		m.textInput.Width = msg.Width - 8
		return m, nil
	case backendMsg:
		if msg.err != nil {
			m.state = stateError
			m.err = msg.err
		} else {
			m.state = stateResult
			m.result = msg.response
		}
		return m, nil
	case spinner.TickMsg:
		if m.state == stateProcessing {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			return m, cmd
		}
	}

	switch m.state {
	case stateMenu:
		m.opList, cmd = m.opList.Update(msg)
	}
	return m, cmd
}

// View renders the TUI based on the current model state.
func (m model) View() string {
	var b strings.Builder
	if m.splash != "" {
		b.WriteString(m.splash)
		b.WriteString("\n")
	}

	switch m.state {
	case stateLicense:
		var viewBuilder strings.Builder
		viewBuilder.WriteString("Please enter your license key to continue.\n\n")
		viewBuilder.WriteString(m.textInput.View())
		viewBuilder.WriteString("\n\n(Press Enter to confirm, q to quit)")
		if m.err != nil {
			viewBuilder.WriteString("\n\n" + errorStyle.Render(m.err.Error()))
		}
		b.WriteString(boxStyle.Render(viewBuilder.String()))
	case stateTextInput:
		var viewBuilder strings.Builder
		viewBuilder.WriteString("Please enter the full path to your model file (.pt, .safetensors, etc.)\n\n")
		viewBuilder.WriteString(m.textInput.View())
		viewBuilder.WriteString("\n\n(Press Enter to confirm, q to quit)")
		if m.err != nil {
			viewBuilder.WriteString("\n\n" + errorStyle.Render(m.err.Error()))
		}
		b.WriteString(boxStyle.Render(viewBuilder.String()))
	case stateMenu:
		b.WriteString(m.renderMenu())
	case stateProcessing:
		b.WriteString(m.renderProcessing())
	case stateResult:
		b.WriteString(m.renderResult())
	case stateError:
		b.WriteString(m.renderError())
	case stateExtraInput:
		var viewBuilder strings.Builder
		viewBuilder.WriteString(fmt.Sprintf("Enter details for: %s\n\n", m.selectedOp.Name))
		viewBuilder.WriteString(m.textInput.View())
		viewBuilder.WriteString("\n\n(Press Enter to confirm, Esc to cancel)")
		if m.err != nil {
			viewBuilder.WriteString("\n\n" + errorStyle.Render(m.err.Error()))
		}
		b.WriteString(boxStyle.Render(viewBuilder.String()))
	default:
		b.WriteString("Unknown state")
	}
	return b.String()
}

// --- TUI Render Helpers ---

// CreateNexaSplash creates a styled splash screen with a random color.
func CreateNexaSplash() string {
	// Select a random color from the palette.
	randomColor := splashColors[rand.Intn(len(splashColors))]

	nexaLines := []string{
		"███╗   ██╗███████╗██╗  ██╗ █████╗     ████████╗ ██████╗  ██████╗ ██╗     ██╗██╗████████╗",
		"████╗  ██║██╔════╝╚██╗██╔╝██╔══██╗    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██║██║╚══██╔══╝",
		"██╔██╗ ██║█████╗   ╚███╔╝ ███████║       ██║   ██║   ██║██║   ██║██║     ██║██║   ██║   ",
		"██║╚██╗██║██╔══╝   ██╔██╗ ██╔══██║       ██║   ██║   ██║██║   ██║██║     ██║██║   ██║   ",
		"██║ ╚████║███████╗██╔╝ ██╗██║  ██║       ██║   ╚██████╔╝╚██████╔╝███████╗██║██║   ██║   ",
		"╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝╚═╝   ╚═╝   ",
	}

	box := lipgloss.NewStyle().
		Border(lipgloss.ThickBorder()).
		BorderForeground(randomColor).
		Padding(1, 4).
		Align(lipgloss.Center)

	style := lipgloss.NewStyle().Foreground(randomColor).Bold(true)

	var styled []string
	for _, line := range nexaLines {
		styled = append(styled, style.Render(line))
	}
	return box.Render(strings.Join(styled, "\n"))
}

func (m model) renderMenu() string {
	var b strings.Builder
	header := headerStyle.Render("🚀 Nexa ToolKit")
	if m.filePath != "" {
		header += "\n" + infoStyle.Render(fmt.Sprintf("File: %s", filepath.Base(m.filePath)))
	}
	b.WriteString(header)
	b.WriteString("\n\n")
	b.WriteString(m.opList.View())
	b.WriteString("\n\n" + footerStyle.Render(fmt.Sprintf("v%s | q: Quit", version)))
	return b.String()
}

func (m model) renderProcessing() string {
	return fmt.Sprintf(
		"\n\n%s %s\n\nProcessing %s...\n\nPress 'q' to quit",
		m.spinner.View(),
		headerStyle.Render("Processing"),
		m.selectedOp.Name,
	)
}

func (m model) renderResult() string {
	if m.result == nil {
		return "No result available"
	}

	var content strings.Builder
	content.WriteString(headerStyle.Render("✓ Operation Completed"))
	content.WriteString("\n\n")

	if m.result.Success {
		content.WriteString(successStyle.Render(m.result.Message))
		content.WriteString(fmt.Sprintf("\nExecution time: %.2fs", m.result.ExecutionTime))

		if m.result.Data != nil {
			content.WriteString("\n\n" + headerStyle.Render("Results:"))
			content.WriteString("\n" + m.formatResultData(m.result.Data))
		}
	} else {
		content.WriteString(errorStyle.Render("Operation failed: " + m.result.Message))
		if m.result.Error != "" {
			content.WriteString("\nError: " + m.result.Error)
		}
	}

	content.WriteString("\n\nPress Enter to return to menu or 'q' to quit")
	content.WriteString("\n\n" + helpStyle.Render("Press 'b' to go back"))
	return boxStyle.Render(content.String())
}

func (m model) renderError() string {
	var content strings.Builder
	content.WriteString(headerStyle.Render("✗ Error"))
	content.WriteString("\n\n")
	content.WriteString(errorStyle.Render(fmt.Sprintf("Failed to execute operation: %v", m.err)))
	content.WriteString("\n\nPress Enter to return to menu or 'q' to quit")
	content.WriteString("\n\n" + helpStyle.Render("Press 'b' to go back"))
	return boxStyle.Render(content.String())
}

// --- TUI Formatting Helpers ---

func (m model) formatResultData(data interface{}) string {
	switch m.selectedOp.Command {
	case "analyze":
		return m.formatAnalysisResult(data)
	case "convert":
		return m.formatConversionResult(data)
	case "quantize":
		return m.formatQuantizationResult(data)
	case "benchmark":
		return m.formatBenchmarkResult(data)
	case "eval":
		return m.formatEvalResult(data)
	case "check":
		return m.formatEnvironmentResult(data)
	case "prune":
		return m.formatPruneResult(data)
	case "merge":
		return m.formatMergeResult(data)
	default:
		if jsonBytes, err := json.MarshalIndent(data, "", "  "); err == nil {
			return string(jsonBytes)
		}
		return fmt.Sprintf("%+v", data)
	}
}

func (m model) formatAnalysisResult(data interface{}) string {
	// Convert to map for easier access
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Sprintf("%+v", data)
	}

	var content strings.Builder

	if format, ok := dataMap["format_type"].(string); ok {
		content.WriteString(fmt.Sprintf("Format: %s\n", format))
	}

	if params, ok := dataMap["num_parameters"].(float64); ok {
		content.WriteString(fmt.Sprintf("Parameters: %s\n", formatNumber(int64(params))))
	}

	if tensors, ok := dataMap["num_tensors"].(float64); ok {
		content.WriteString(fmt.Sprintf("Tensors: %d\n", int(tensors)))
	}

	if memSize, ok := dataMap["memory_size"].(float64); ok {
		content.WriteString(fmt.Sprintf("Memory Size: %s\n", formatBytes(int64(memSize))))
	}

	if fileSize, ok := dataMap["file_size"].(float64); ok {
		content.WriteString(fmt.Sprintf("File Size: %s\n", formatBytes(int64(fileSize))))
	}

	if arch, ok := dataMap["architecture"].(string); ok {
		content.WriteString(fmt.Sprintf("Architecture: %s\n", arch))
	}

	if dtype, ok := dataMap["dtype"].(string); ok {
		content.WriteString(fmt.Sprintf("Data Type: %s\n", dtype))
	}

	return content.String()
}

func (m model) formatConversionResult(data interface{}) string {
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Sprintf("%+v", data)
	}

	var content strings.Builder

	if input, ok := dataMap["input_file"].(string); ok {
		content.WriteString(fmt.Sprintf("Input: %s\n", filepath.Base(input)))
	}

	if output, ok := dataMap["output_file"].(string); ok {
		content.WriteString(fmt.Sprintf("Output: %s\n", filepath.Base(output)))
	}

	if sourceFormat, ok := dataMap["source_format"].(string); ok {
		if targetFormat, ok := dataMap["target_format"].(string); ok {
			content.WriteString(fmt.Sprintf("Conversion: %s → %s\n", sourceFormat, targetFormat))
		}
	}

	if outputSize, ok := dataMap["output_size"].(float64); ok {
		content.WriteString(fmt.Sprintf("Output Size: %s\n", formatBytes(int64(outputSize))))
	}

	return content.String()
}

func (m model) formatQuantizationResult(data interface{}) string {
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Sprintf("%+v", data)
	}

	var content strings.Builder

	if method, ok := dataMap["method"].(string); ok {
		content.WriteString(fmt.Sprintf("Method: %s\n", method))
	}

	if originalSize, ok := dataMap["original_size"].(float64); ok {
		content.WriteString(fmt.Sprintf("Original Size: %s\n", formatBytes(int64(originalSize))))
	}

	if quantizedSize, ok := dataMap["quantized_size"].(float64); ok {
		content.WriteString(fmt.Sprintf("Quantized Size: %s\n", formatBytes(int64(quantizedSize))))
	}

	if ratio, ok := dataMap["compression_ratio"].(float64); ok {
		content.WriteString(fmt.Sprintf("Compression Ratio: %.2fx\n", ratio))
	}

	if saved, ok := dataMap["space_saved"].(float64); ok {
		content.WriteString(fmt.Sprintf("Space Saved: %s\n", formatBytes(int64(saved))))
	}

	return content.String()
}

func (m model) formatPruneResult(data interface{}) string {
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Sprintf("%+v", data)
	}

	var content strings.Builder

	if amount, ok := dataMap["pruning_amount"].(float64); ok {
		content.WriteString(fmt.Sprintf("Pruning Amount: %.0f%%\n", amount*100))
	}

	if input, ok := dataMap["input_file"].(string); ok {
		content.WriteString(fmt.Sprintf("Input: %s\n", filepath.Base(input)))
	}

	if output, ok := dataMap["output_file"].(string); ok {
		content.WriteString(fmt.Sprintf("Output: %s\n", filepath.Base(output)))
	}

	return content.String()
}

func (m model) formatMergeResult(data interface{}) string {
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Sprintf("%+v", data)
	}

	var content strings.Builder

	if output, ok := dataMap["output_file"].(string); ok {
		content.WriteString(fmt.Sprintf("Output: %s\n", filepath.Base(output)))
	}
	if num, ok := dataMap["models_merged"].(float64); ok {
		content.WriteString(fmt.Sprintf("Models Merged: %d\n", int(num)))
	}
	if size, ok := dataMap["output_size"].(float64); ok {
		content.WriteString(fmt.Sprintf("Output Size: %s\n", formatBytes(int64(size))))
	}

	return content.String()
}

func (m model) formatBenchmarkResult(data interface{}) string {
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Sprintf("%+v", data)
	}

	var content strings.Builder

	if device, ok := dataMap["device"].(string); ok {
		content.WriteString(fmt.Sprintf("Device: %s\n", device))
	}
	if numRuns, ok := dataMap["num_runs"].(float64); ok {
		content.WriteString(fmt.Sprintf("Runs: %d\n", int(numRuns)))
	}
	if shape, ok := dataMap["input_shape"].([]interface{}); ok {
		content.WriteString(fmt.Sprintf("Input Shape: %v\n", shape))
	}
	if latency, ok := dataMap["avg_latency_ms"].(float64); ok {
		content.WriteString(fmt.Sprintf("Avg Latency: %.2f ms\n", latency))
	}
	if throughput, ok := dataMap["throughput_fps"].(float64); ok {
		content.WriteString(fmt.Sprintf("Throughput: %.2f FPS\n", throughput))
	}

	return content.String()
}

func (m model) formatEvalResult(data interface{}) string {
	// Eval result is a dict, so just pretty-print it as JSON.
	if jsonBytes, err := json.MarshalIndent(data, "", "  "); err == nil {
		return string(jsonBytes)
	}
	return fmt.Sprintf("%+v", data)
}

func (m model) formatEnvironmentResult(data interface{}) string {
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Sprintf("%+v", data)
	}

	var content strings.Builder
	content.WriteString("Dependencies:\n")

	if deps, ok := dataMap["dependencies"].(map[string]interface{}); ok {
		for name, info := range deps {
			if depInfo, ok := info.(map[string]interface{}); ok {
				if available, ok := depInfo["available"].(bool); ok {
					status := "❌"
					if available {
						status = "✅"
					}
					content.WriteString(fmt.Sprintf("  %s %s", status, name))
					if version, ok := depInfo["version"].(string); ok {
						content.WriteString(fmt.Sprintf(" (%s)", version))
					}
					content.WriteString("\n")
				}
			}
		}
	}

	return content.String()
}

// --- Backend Communication ---

func (m model) runBackendOperation(operation string) tea.Cmd {
	return func() tea.Msg {
		// Prepare backend request payload
		payload := make(map[string]interface{})

		switch operation {
		case "merge":
			payload["file_paths"] = m.filePaths
		case "benchmark":
			payload["file_path"] = m.filePath
			shape, err := parseShape(m.benchmarkInputShape)
			if err != nil {
				return backendMsg{err: err}
			}
			payload["input_shape"] = shape
			payload["num_runs"] = 50 // Default runs for TUI
		case "eval":
			payload["model_path"] = m.filePath
			payload["eval_script_path"] = m.evalScriptPath
		case "convert":
			payload["file_path"] = m.filePath
			payload["target_format"] = "safetensors" // This should be configurable
		case "quantize":
			payload["file_path"] = m.filePath
			payload["method"] = "fp16" // This should be configurable
		case "prune":
			payload["file_path"] = m.filePath
			payload["amount"] = 0.2 // Default pruning amount
			payload["pruning_type"] = "unstructured"
		default:
			if operation != "check" {
				payload["file_path"] = m.filePath
			}
		}

		// Execute backend command
		response, err := executeBackend(operation, payload, m.licenseKey)
		return backendMsg{response: response, err: err}
	}
}

// executeBackend sends an HTTP request to the Python backend server.
func executeBackend(operation string, payload map[string]interface{}, licenseKey string) (*BackendResponse, error) {
	endpoint := fmt.Sprintf("%s/%s", backendURL, operation)
	method := "POST"
	if operation == "check" {
		method = "GET"
	}

	var reqBody io.Reader
	if method == "POST" {
		jsonBytes, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %v", err)
		}
		reqBody = bytes.NewBuffer(jsonBytes)
	}

	req, err := http.NewRequest(method, endpoint, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-License-Key", licenseKey)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("backend request failed: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errDetail ErrorDetail
		if json.Unmarshal(body, &errDetail) == nil {
			return &errDetail.Detail, nil
		}
		return nil, fmt.Errorf("backend returned non-200 status: %s\nBody: %s", resp.Status, string(body))
	}

	var response BackendResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse backend response: %v\nOutput: %s", err, string(body))
	}

	return &response, nil
}

// --- Server Management ---

// startBackendServer starts the Python FastAPI backend server as a subprocess.
func startBackendServer(ctx context.Context) (*exec.Cmd, error) {
	pythonCmd := findPythonCommand()
	if pythonCmd == "" {
		return nil, fmt.Errorf("python interpreter not found")
	}
	backendScript := findBackendScript()
	if backendScript == "" {
		return nil, fmt.Errorf("Backend.py script not found")
	}

	cmd := exec.CommandContext(ctx, pythonCmd, backendScript)
	// To prevent the backend from being killed if the parent Go process is killed unexpectedly
	if runtime.GOOS != "windows" {
		// This is not available on Windows
		// cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}

	err := cmd.Start()
	if err != nil {
		return nil, fmt.Errorf("failed to start backend server: %w", err)
	}

	// Health check to ensure server is up
	for i := 0; i < 10; i++ {
		time.Sleep(500 * time.Millisecond)
		if _, err := http.Get(backendURL); err == nil {
			fmt.Println("Backend server started successfully.")
			return cmd, nil
		}
	}

	cmd.Process.Kill()
	return nil, fmt.Errorf("backend server failed to start in time")
}

// --- Utility Functions ---

func findPythonCommand() string {
	candidates := []string{"python3", "python", "py"}
	for _, cmd := range candidates {
		if _, err := exec.LookPath(cmd); err == nil {
			return cmd
		}
	}
	return ""
}

func findBackendScript() string {
	candidates := []string{
		"Backend.py",
		"backend.py",
	}

	// When running as a compiled executable, the script might be in the same directory
	exePath, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exePath)
		candidates = append(candidates, filepath.Join(dir, "Backend.py"))
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			absPath, _ := filepath.Abs(path)
			return absPath
		}
	}
	return ""
}

// formatBytes converts bytes to a human-readable string.
func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// formatNumber converts a large number to a human-readable string (K, M, B).
func formatNumber(n int64) string {
	if n < 1000 {
		return fmt.Sprintf("%d", n)
	}
	if n < 1000000 {
		return fmt.Sprintf("%.1fK", float64(n)/1000)
	}
	if n < 1000000000 {
		return fmt.Sprintf("%.1fM", float64(n)/1000000)
	}
	return fmt.Sprintf("%.1fB", float64(n)/1000000000)
}

func parseShape(shapeStr string) ([]int, error) {
	parts := strings.Split(shapeStr, ",")
	shape := make([]int, len(parts))
	for i, part := range parts {
		val, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("invalid shape component: %s", part)
		}
		shape[i] = val
	}
	return shape, nil
}

// isValidModelFile checks if a file path exists and has a valid model extension.
func isValidModelFile(filePath string) bool {
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return false
	}

	ext := strings.ToLower(filepath.Ext(filePath))
	validExts := []string{".pt", ".pth", ".safetensors", ".bin"}

	for _, validExt := range validExts {
		if ext == validExt {
			return true
		}
	}
	return false
}

// --- CLI Command Logic ---

func runTUI(filePath string) {
	if filePath != "" && !isValidModelFile(filePath) {
		fmt.Printf("Error: File not found or invalid: %s\n", filePath)
		os.Exit(1)
	}

	p := tea.NewProgram(initialModel(filePath), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error running TUI: %v\n", err)
		os.Exit(1)
	}
}

func runDirectOperation(operation string, fileArgs ...string) {
	payload := make(map[string]interface{})

	switch operation {
	case "merge":
		if len(fileArgs) < 1 {
			fmt.Println("Error: merge command requires a comma-separated list of model files.")
			os.Exit(1)
		}
		payload["file_paths"] = strings.Split(fileArgs[0], ",")
	case "benchmark":
		if len(fileArgs) < 1 {
			fmt.Println("Error: benchmark command requires a model file.")
			os.Exit(1)
		}
		payload["file_path"] = fileArgs[0]
		shape, err := parseShape(benchmarkShapeFlag)
		if err != nil {
			fmt.Printf("Error: invalid shape format: %v\n", err)
			os.Exit(1)
		}
		payload["input_shape"] = shape
		payload["num_runs"] = benchmarkRunsFlag
	case "eval":
		if len(fileArgs) < 2 {
			fmt.Println("Error: eval command requires a model file and a script file.")
			os.Exit(1)
		}
		payload["model_path"] = fileArgs[0]
		payload["eval_script_path"] = fileArgs[1]
	default:
		if len(fileArgs) > 0 {
			payload["file_path"] = fileArgs[0]
		}
	}

	if outputFlag != "" {
		payload["output_path"] = outputFlag
	}
	if formatFlag != "" {
		payload["target_format"] = formatFlag
	}
	if methodFlag != "" {
		payload["method"] = methodFlag
	}
	if pruneTypeFlag != "" {
		payload["pruning_type"] = pruneTypeFlag
	}
	if pruneAmountFlag > 0 {
		payload["amount"] = pruneAmountFlag
	}

	fmt.Printf("Executing %s...\n", operation)
	start := time.Now()

	// For direct CLI operations, we need to manage the server lifecycle.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	serverCmd, err := startBackendServer(ctx)
	if err != nil {
		fmt.Printf("Error starting backend: %v\n", err)
		os.Exit(1)
	}
	defer serverCmd.Process.Kill()

	licenseKey := os.Getenv(licenseEnvVar)
	if licenseKey == "" {
		fmt.Println("Error: NEXA_LICENSE_KEY environment variable not set.")
		os.Exit(1)
	}

	response, err := executeBackend(operation, payload, licenseKey)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		os.Exit(1)
	}

	duration := time.Since(start)

	if response.Success {
		fmt.Printf("✅ %s\n", response.Message)
		fmt.Printf("⏱️  Execution time: %.2fs\n", duration.Seconds())

		if verboseFlag && response.Data != nil {
			fmt.Println("\nResults:")
			if jsonBytes, err := json.MarshalIndent(response.Data, "", "  "); err == nil {
				fmt.Println(string(jsonBytes))
			}
		}
	} else {
		fmt.Printf("❌ Operation failed: %s\n", response.Message)
		if response.Error != "" {
			fmt.Printf("Error details: %s\n", response.Error)
		}
		os.Exit(1)
	}
}

func runRootCommand(cmd *cobra.Command, args []string) {
	if len(args) == 0 {
		// Start server and run TUI without a file
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		serverCmd, err := startBackendServer(ctx)
		if err != nil {
			fmt.Printf("Error starting backend: %v\n", err)
			os.Exit(1)
		}
		defer serverCmd.Process.Kill()
		runTUI("")
		return
	}

	// Handle cases with arguments
	filePath := args[0]
	if !isValidModelFile(filePath) {
		// Argument is not a file, assume it's a command like "check"
		runDirectOperation(filePath)
	} else {
		// Argument is a file
		if len(args) > 1 {
			// Direct command: nexa-toolkit file.pt operation
			runDirectOperation(args[1], filePath)
		} else {
			// TUI mode with file
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			serverCmd, err := startBackendServer(ctx)
			if err != nil {
				fmt.Printf("Error starting backend: %v\n", err)
				os.Exit(1)
			}
			defer serverCmd.Process.Kill()
			runTUI(filePath)
		}
	}
}

// --- CLI Command Definitions ---

var (
	outputFlag         string
	formatFlag         string
	methodFlag         string
	verboseFlag        bool
	pruneTypeFlag      string
	pruneAmountFlag    float64
	benchmarkShapeFlag string
	benchmarkRunsFlag  int
)

var rootCmd = &cobra.Command{
	Use:   "nexa-toolkit [model_file]",
	Short: "Nexa ToolKit - High-performance model processing tool",
	Long: `Nexa ToolKit combines a Go frontend with a Python backend for ML model operations.
It can be run in an interactive TUI mode or via direct CLI commands.`,
	Run: runRootCommand,
}

var analyzeCmd = &cobra.Command{
	Use:   "analyze [model_file]",
	Short: "Analyze model statistics and metadata",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("analyze", args[0])
	},
}

var convertCmd = &cobra.Command{
	Use:   "convert [model_file]",
	Short: "Convert between model formats",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("convert", args[0])
	},
}

var quantizeCmd = &cobra.Command{
	Use:   "quantize [model_file]",
	Short: "Quantize model weights",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("quantize", args[0])
	},
}

var pruneCmd = &cobra.Command{
	Use:   "prune [model_file]",
	Short: "Prune model parameters",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("prune", args[0])
	},
}

var mergeCmd = &cobra.Command{
	Use:   "merge [model_file1,model_file2,...]",
	Short: "Merge multiple models",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("merge", args[0])
	},
}

var benchmarkCmd = &cobra.Command{
	Use:   "benchmark [model_file]",
	Short: "Benchmark model performance",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("benchmark", args[0])
	},
}

var evalCmd = &cobra.Command{
	Use:   "eval [model_file] [script_file]",
	Short: "Run a custom evaluation script",
	Args:  cobra.ExactArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("eval", args[0], args[1])
	},
}

var checkCmd = &cobra.Command{
	Use:   "check",
	Short: "Check environment and dependencies",
	Args:  cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		runDirectOperation("check", "")
	},
}

// init sets up the CLI commands and flags.
func init() {
	// Seed the random number generator for the splash screen.
	rand.Seed(time.Now().UnixNano())

	rootCmd.PersistentFlags().StringVarP(&outputFlag, "output", "o", "", "Output file path")
	rootCmd.PersistentFlags().StringVarP(&formatFlag, "format", "f", "safetensors", "Target format (safetensors, pytorch)")
	rootCmd.PersistentFlags().StringVarP(&methodFlag, "method", "m", "fp16", "Quantization method (fp16, bf16, int8)")
	rootCmd.PersistentFlags().BoolVarP(&verboseFlag, "verbose", "v", false, "Verbose output")

	pruneCmd.Flags().StringVar(&pruneTypeFlag, "type", "unstructured", "Pruning type (unstructured, structured)")
	pruneCmd.Flags().Float64Var(&pruneAmountFlag, "amount", 0.2, "Pruning amount (0.0 to 1.0)")

	benchmarkCmd.Flags().StringVar(&benchmarkShapeFlag, "shape", "1,3,224,224", "Input shape for benchmark (e.g., 1,3,224,224)")
	benchmarkCmd.Flags().IntVar(&benchmarkRunsFlag, "runs", 50, "Number of runs for benchmark")

	// Add subcommands
	rootCmd.AddCommand(analyzeCmd)
	rootCmd.AddCommand(convertCmd)
	rootCmd.AddCommand(quantizeCmd)
	rootCmd.AddCommand(pruneCmd)
	rootCmd.AddCommand(mergeCmd)
	rootCmd.AddCommand(benchmarkCmd)
	rootCmd.AddCommand(evalCmd)
	rootCmd.AddCommand(checkCmd)
}

// main is the entry point of the application.
func main() {
    // Block running from Go's temp build directory and guide user to build properly.
    if exe, err := os.Executable(); err == nil {
        if strings.HasPrefix(strings.ToLower(exe), strings.ToLower(os.TempDir())) {
            fmt.Fprintln(os.Stderr,
                "Error: Access denied for temporary build.\n"+
                "Please compile the application first:\n"+
                "  go build -o nexa-toolkit.exe main.go\n"+
                "Then run the generated `nexa-toolkit.exe`.")
            os.Exit(1)
        }
    }

    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
