import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def sift_mosaic(im1, im2):
    """
    Compute a mosaic of two images using SIFT features and RANSAC.
    """
    # Convert to grayscale if needed
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) if im1.ndim == 3 else im1
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) if im2.ndim == 3 else im2

    # --- Detect SIFT keypoints and descriptors ---
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # --- Match descriptors using FLANN ---
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # --- Lowe's ratio test ---
    good_matches = [m for m,n in matches if m.distance < 0.7 * n.distance]
    num_matches = len(good_matches)
    print(f"Tentative matches: {num_matches}")

    if num_matches < 4:
        raise ValueError("Not enough matches to compute homography.")

    # --- Extract matched points ---
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # --- Compute homography with RANSAC ---
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(H, "homogeneous")
    inliers = mask.ravel().tolist()
    print(f"Inlier matches: {sum(inliers)} ({100*sum(inliers)/num_matches:.2f}%)")

    # --- Draw inlier matches ---
    img_matches = cv2.drawMatches(
        im1, kp1, im2, kp2, good_matches, None,
        matchColor=(0,255,0),
        singlePointColor=None,
        matchesMask=inliers,
        flags=2
    )
    plt.figure(figsize=(15,7))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Inlier matches after RANSAC")
    plt.axis('off')
    plt.show()

    # --- Create mosaic ---
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    corners_im2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    corners_im1 = cv2.perspectiveTransform(np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2), H)

    all_corners = np.concatenate((corners_im1, corners_im2), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    H_translation = np.array([[1,0,translation[0]],
                              [0,1,translation[1]],
                              [0,0,1]])

    mosaic = cv2.warpPerspective(im1, H_translation @ H, (xmax-xmin, ymax-ymin))
    mosaic[translation[1]:h2+translation[1], translation[0]:w2+translation[0]] = im2

    plt.figure(figsize=(15,10))
    plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    plt.title("Mosaic")
    plt.axis('off')
    plt.show()

    return mosaic

# --- Main function to parse command-line arguments ---
def main():
    parser = argparse.ArgumentParser(description="SIFT-based image mosaic")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    args = parser.parse_args()

    # Read images
    im1 = cv2.imread(args.image1)
    im2 = cv2.imread(args.image2)

    # Check if images are loaded
    if im1 is None or im2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    # Compute mosaic
    sift_mosaic(im1, im2)

# --- Entry point ---
if __name__ == "__main__":
    main()
