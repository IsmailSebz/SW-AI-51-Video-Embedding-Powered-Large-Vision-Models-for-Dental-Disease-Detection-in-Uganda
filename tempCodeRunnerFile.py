    # for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
    #     x1, y1, x2, y2, conf, cls = result[:6]
    #     label = f'{model.names[cls]} {conf:.2f}'
        
    #     # Draw bounding box and label on the frame
    #     if conf > 0.5: 
    #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)  # Bounding box
    #             # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    