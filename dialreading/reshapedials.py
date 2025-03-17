# Thanks Copilot
import cv2
import numpy as np
import os
import json

# Global variable for clicked points
points = []

def get_points_on_image(img):
    """Allows the user to click points on the image."""
    global points
    points = []  # Reset points

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)  # Draw point
            cv2.imshow('Image', temp_img)

    # Display image for point selection
    temp_img = img.copy()
    cv2.imshow('Image', temp_img)
    cv2.setMouseCallback('Image', click_event)
    print("Click at least 5 points on the image. Press 'q' when done.")

    while True:
        key = cv2.waitKey(1)
        if key == ord('q') and len(points) >= 5:
            break
        elif key == ord('q'):
            print("Please click at least 5 points!")
    cv2.destroyAllWindows()
    return points

def fit_ellipse_and_calculate_scaling(points):
    """
    Fits an ellipse to the points and determines major/minor axes,
    as well as their orientation.
    """
    points = np.array(points, dtype=np.int32)
    if len(points) < 5:
        raise ValueError("At least 5 points are required to fit an ellipse.")

    # Fit an ellipse to the points
    ellipse = cv2.fitEllipse(points)
    center, axes, angle = ellipse
    width, height = axes

    # Determine the orientation of the major axis
    if width > height:
        major_axis = width
        minor_axis = height
        orientation = "x-axis" if abs(angle) < 45 or abs(angle) > 135 else "y-axis"
    else:
        major_axis = height
        minor_axis = width
        orientation = "y-axis" if abs(angle) < 45 or abs(angle) > 135 else "x-axis"

    print(f"Fitted Ellipse: Major Axis = {major_axis:.2f}, Minor Axis = {minor_axis:.2f}, Orientation = {orientation}")

    # Calculate scaling factors
    scale_x = minor_axis / major_axis if orientation == "x-axis" else 1
    scale_y = minor_axis / major_axis if orientation == "y-axis" else 1

    print(f"Correct Scaling Factors: scale_x = {scale_x}, scale_y = {scale_y}")

    return scale_x, scale_y, ellipse

def apply_scaling(img, scale_x, scale_y):
    """Applies correct scaling factors."""
    # Rescale the image
    rescaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return rescaled_img

def verify_rescaled_points(points):
    """Calculates error based on the rescaled points."""
    points = np.array(points)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    major_axis = np.max(distances) * 2
    minor_axis = np.min(distances) * 2
    error = abs(major_axis - minor_axis) / max(major_axis, minor_axis)
    print(f"Verification: Major Axis = {major_axis}, Minor Axis = {minor_axis}, Error = {error}")
    return error, major_axis, minor_axis

def save_progress(output_dir, iteration, img, scale_x, scale_y):
    """Saves the current progress: image and scaling factors."""
    # Save the image
    output_file = os.path.join(output_dir, f"progress_iteration_{iteration}.png")
    cv2.imwrite(output_file, img)
    print(f"Progress saved: {output_file}")

    # Save the scaling factors
    scaling_file = os.path.join(output_dir, "scaling_factors.json")
    with open(scaling_file, 'w') as f:
        json.dump({"iteration": iteration, "scale_x": scale_x, "scale_y": scale_y}, f)
    print(f"Scaling factors saved: {scaling_file}")

def iterative_scaling(img, output_dir, tolerance=0.02, max_iterations=20):
    """Performs iterative scaling to minimize error and make the ellipse circular."""
    points = get_points_on_image(img)  # Initial points
    iteration = 1  # Local iteration variable
    scaling_coefficient = 0.5  # Limits sensitivity of scaling adjustments
    flatline_threshold = 0.001  # Detect flatlining
    minpercent = 0.01
    maxpercent = 0.2
    previous_error = None

    # Initialize cumulative scaling factors
    overall_scale_x = 1.0
    overall_scale_y = 1.0

    while iteration <= max_iterations:
        print(f"\nIteration {iteration}: Rescaling and recalculating...")

        # Fit ellipse and calculate scaling factors
        _, _, ellipse = fit_ellipse_and_calculate_scaling(points)
        major_axis, minor_axis = ellipse[1]

        # Calculate error and adjustment factor
        error = abs(major_axis - minor_axis) / max(major_axis, minor_axis)
        adjustment_factor = 1 + (error * scaling_coefficient)
        adjustment_factor = min(max( (1+minpercent), adjustment_factor), (1 + maxpercent))
        print(f"Iteration {iteration}: Major Axis = {major_axis:.2f}, Minor Axis = {minor_axis:.2f}, Error = {error:.4f}, Adjustment Factor = {adjustment_factor:.4f}")

        # Stop if error is within tolerance
        if error <= tolerance:
            print("Success! The rescaled image forms a circle within the acceptable error tolerance.")
            save_progress(output_dir, iteration, img, overall_scale_x, overall_scale_y)
            return img

        # Smooth scaling adjustments
        if major_axis > minor_axis:
            scale_x = 1 / adjustment_factor  # Compress the major axis
            scale_y = adjustment_factor      # Stretch the minor axis
        else:
            scale_x = adjustment_factor      # Stretch the minor axis
            scale_y = 1 / adjustment_factor  # Compress the major axis

        # Update cumulative scaling factors
        overall_scale_x *= scale_x
        overall_scale_y *= scale_y

        # Apply scaling
        img = apply_scaling(img, scale_x, scale_y)
        points = get_points_on_image(img)  # Update points from the rescaled image

        # Detect flatlining and reduce scaling further
        if previous_error is not None and abs(previous_error - error) < flatline_threshold:
            scaling_coefficient *= 0.9  # Reduce sensitivity when improvements flatline
            print(f"Flatline detected. Reducing scaling sensitivity: {scaling_coefficient:.4f}")

        # Save progress if the user presses 's'
        print("Press 's' to stop and save progress, or press any other key to continue.")
        key = cv2.waitKey(5000)  # Wait 5 seconds for user input
        if key == ord('s'):
            print("Stopping and saving progress...")
            save_progress(output_dir, iteration, img, overall_scale_x, overall_scale_y)
            return img

        previous_error = error
        iteration += 1

    print("Max iterations reached. The process did not converge.")
    save_progress(output_dir, iteration, img, overall_scale_x, overall_scale_y)
    return img


def generate_incremental_filename(output_dir, base_name="p"):
    """Generates the next available numeric filename."""
    # List all files in the output directory
    existing_files = os.listdir(output_dir)
    
    # Find all files that match the base name format
    existing_numbers = []
    for file_name in existing_files:
        if file_name.startswith(base_name) and file_name.endswith(".png"):
            try:
                # Extract the numeric part of the file name
                number = int(file_name[len(base_name) + 1:-4])  # Exclude base_name_ and .png
                existing_numbers.append(number)
            except ValueError:
                continue

    # Determine the next available number
    next_number = max(existing_numbers, default=0) + 1
    return os.path.join(output_dir, f"{base_name}_{next_number}.png")

if __name__ == "__main__":
    path = 'data/frames/'
    img_path = path + "t1.png"  # Path to your image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image at {img_path}")
        exit()

    # Create the output directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    rescaled_img = iterative_scaling(img, path)
    output_file = generate_incremental_filename(path)
    cv2.imwrite(output_file, rescaled_img)
    print(f"Final rescaled image saved to {output_file}")

    cv2.imshow("Final Rescaled Image", rescaled_img)
    cv2.wait